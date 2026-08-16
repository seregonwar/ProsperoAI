/*
 * ProsperoAI — PS5 GPU page-table (gvmspace) repair interface.
 */

#ifndef PAI_PLATFORM_PS5_GVMSPACE_H
#define PAI_PLATFORM_PS5_GVMSPACE_H

#include <pai/error.h>

#include <stdint.h>

/* Select the repair rehearsal mode (0 probe / 1 no-op write / 2 full). */
void pai_gvmspace_set_mode(int mode);

/*
 * Repair the GPU mapping for `gpu_va` (2 MB granularity) after the
 * walk has been VALIDATED against a kernel-mapped reference VA.
 * Writes a 2 MB leaf PDE with the reference flags and our physical
 * frame, then verifies. Requires the layout + a captured reference PDE.
 */
pai_status_t pai_gvmspace_repair(uint64_t gpu_va, uint64_t phys);

/*
 * Retrieve the layout values the diagnostic derived (read-only):
 * the GPU pml4 physical address and the direct-map base.
 * Returns -1 when the layout is not known yet.
 */
int pai_gvmspace_layout(uint64_t *out_pml4_phys, intptr_t *out_dmap);

/*
 * Read-only probe: walk a candidate GPU pml4 (physical address) for
 * `va` through the direct map and report the resulting PDE. No writes.
 */
pai_status_t pai_gvmspace_probe(uint64_t pml4_phys, uint64_t va,
                                intptr_t dmap_base);

/*
 * Read-only: scan physical memory (via the direct map) for magic,
 * sampling step-spaced 16-byte windows. Logs up to max_hits.
 * Returns the number of hits.
 */
int pai_phys_scan(uint32_t magic, uint64_t start, uint64_t end,
                  uint64_t step, int max_hits);

/*
 * Read-only: scan physical memory (via the direct map) for magic,
 * sampling step-spaced 16-byte windows. Logs up to max_hits.
 * Returns the number of hits.
 */
int pai_phys_scan(uint32_t magic, uint64_t start, uint64_t end,
                  uint64_t step, int max_hits);

/*
 * Read-only: resolve the CPU physical address of a kernel-mapped VA by
 * walking the process CPU page tables (CR3 discovered from the inline
 * vm_pmap). Returns -1 on failure.
 */
int pai_cpu_phys_of_va(uint64_t va, uint64_t *out_phys);

/*
 * Read-only: dump the first 4 words of a raw CPU physical address
 * through the direct map. Used to locate where GPU writes actually
 * land (compare with the syscall phys values).
 */
int pai_gvmspace_dump_phys(uint64_t cpu_phys, uint32_t words[4]);

/*
 * Read-only: dump the first 4 words of the physical page the GPU PDE
 * for a points at. Used to detect kernel-side staging (the VA maps
 * to a separate physical page that the kernel copies lazily). No
 * writes; returns the page content words.
 */
int pai_gvmspace_dump_pde_page(uint64_t va, uint32_t words[4]);

/*
 * Read-only layout diagnostic: dumps the process vmspace pointer
 * landscape and the candidate pmap structs to the log. No kernel
 * writes are performed — safe to run on a live console.
 */
pai_status_t pai_gvmspace_diag(void);

/*
 * Ensure the GPU page tables map `gpu_va` to `phys` (2 MB granularity).
 * Scans kernel data for the process gvmspace, walks the GPU page
 * tables through the direct map and patches the PDE physical frame,
 * preserving flag bits. Kernel rw is required.
 *
 * DANGEROUS: only call with offsets/layout verified for the target
 * firmware. Disabled in the backend until the 9.40 layout is known.
 */
pai_status_t pai_gvmspace_fix(uint64_t gpu_va, uint64_t phys, uint64_t size);

#endif /* PAI_PLATFORM_PS5_GVMSPACE_H */
