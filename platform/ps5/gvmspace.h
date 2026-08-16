/*
 * ProsperoAI — PS5 GPU page-table (gvmspace) repair interface.
 */

#ifndef PAI_PLATFORM_PS5_GVMSPACE_H
#define PAI_PLATFORM_PS5_GVMSPACE_H

#include <pai/error.h>

#include <stdint.h>

/*
 * Ensure the GPU page tables map `gpu_va` to `phys` (2 MB granularity).
 * Scans kernel data for the process gvmspace, walks the GPU page
 * tables through the direct map and patches the PDE physical frame,
 * preserving flag bits. Kernel rw is required.
 */
pai_status_t pai_gvmspace_fix(uint64_t gpu_va, uint64_t phys, uint64_t size);

#endif /* PAI_PLATFORM_PS5_GVMSPACE_H */
