/*
 * ProsperoAI — PS5 GPU page-table (gvmspace) repair interface.
 */

#ifndef PAI_PLATFORM_PS5_GVMSPACE_H
#define PAI_PLATFORM_PS5_GVMSPACE_H

#include <pai/error.h>

#include <stdint.h>

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
