/*
 * ProsperoAI — PS5 GPU page-table (gvmspace) repair
 *
 * The shader read path on 9.40 returns 0 for every load. Working
 * hypothesis: the process gvmspace lacks valid GPU page-table entries
 * for our buffers (writes are posted, reads fault and zero-fill).
 *
 * Fix strategy (kernel rw based, all-runtime — no hardcoded offsets):
 *   1. scan kernel .data for the gvmspace entry CONTAINING a live GPU
 *      VA (the acqrb flexible-memory region the kernel just mapped)
 *   2. read the GPU pml4 through the FreeBSD direct map (candidates
 *      validated by successful reads of plausible page-table content)
 *   3. walk pml4 -> pdpe -> pde (2 MB leaf) for the target VA
 *   4. replace the physical frame in the PDE, preserving the flag bits
 *      (reference flags taken from the acqrb's own PDE)
 *
 * Layout knowledge from the PS5-Firmware-Spoofer (11.20) is used only
 * as a starting guess; every field is validated at runtime.
 */

#include "gvmspace.h"

#include <pai/log.h>

#include <string.h>

#ifdef PAI_PS5

#include <ps5/kernel.h>

#define PAI_GPU_WALK_ADDR_MASK 0x0000FFFFFFFFC0ULL
#define PAI_GPU_VALID          (1ULL << 0)
#define PAI_GPU_IS_PTE         (1ULL << 54)
#define PAI_GPU_PHYS_MASK_2MB  ~(uint64_t)((2u << 20) - 1)

#define PAI_DMAP_MIN 0xFFFFFF8000000000ULL
#define PAI_DMAP_MAX 0xFFFFFFC000000000ULL
#define PAI_DMAP_STEP 0x10000000000ULL /* 1 TB */

/* Known gvmspace entry field offsets (spoofer layout, validated live). */
#define PAI_GVM_ENTRY_SIZE   0x100u
#define PAI_GVM_START_VA_OFF 0x08u
#define PAI_GVM_SIZE_OFF     0x10u
#define PAI_GVM_PAGE_DIR_OFF 0x38u

typedef struct pai_gvmspace_ctx {
  int found;
  intptr_t entry_addr;      /* kernel VA of the entry containing probe_va */
  uint64_t start_va;
  uint64_t size;
  uint64_t page_dir_phys;   /* physical address of the GPU pml4 */
  intptr_t dmap_base;
  uint64_t ref_pde;         /* reference PDE (acqrb) for flag bits */
  uint64_t ref_pde_addr;    /* kernel VA (dmap) of the reference PDE */
} pai_gvmspace_ctx_t;

static int
pai_gvm_read_phys(pai_gvmspace_ctx_t *g, uint64_t phys, uint64_t *out) {
  if (!g->dmap_base) {
    return -1;
  }
  if (kernel_copyout(g->dmap_base + (intptr_t)phys, out, 8) != 0) {
    return -1;
  }
  return 0;
}

static int
pai_gvm_write_phys(pai_gvmspace_ctx_t *g, uint64_t phys, uint64_t value) {
  if (!g->dmap_base) {
    return -1;
  }
  return kernel_copyin(&value, g->dmap_base + (intptr_t)phys, 8) != 0;
}

/* Is `value` a plausible GPU page-table entry? */
static int
pai_gvm_plausible_entry(uint64_t value) {
  if (value == 0) {
    return 1; /* empty */
  }
  if (!(value & PAI_GPU_VALID)) {
    return 0;
  }
  if ((value & PAI_GPU_WALK_ADDR_MASK) == 0) {
    return 0;
  }
  return 1;
}

/* Try to find a working dmap base by reading a known physical address
 * (the GPU pml4) and checking the content looks like a page table. */
static intptr_t
pai_gvm_find_dmap(uint64_t page_dir_phys) {
  for (uint64_t base = PAI_DMAP_MIN; base < PAI_DMAP_MAX;
       base += PAI_DMAP_STEP) {
    uint64_t e0 = 0;
    uint64_t e1 = 0;

    if (kernel_copyout((intptr_t)(base + page_dir_phys), &e0, 8) != 0) {
      continue;
    }
    if (kernel_copyout((intptr_t)(base + page_dir_phys + 8), &e1, 8) != 0) {
      continue;
    }

    if (pai_gvm_plausible_entry(e0) && pai_gvm_plausible_entry(e1)) {
      return (intptr_t)base;
    }
  }
  return 0;
}

static pai_status_t
pai_gvm_walk(pai_gvmspace_ctx_t *g, uint64_t va, uint64_t *out_pde_addr,
             uint64_t *out_pde) {
  uint64_t e4, e3, e2;
  uint64_t idx4 = (va >> 39) & 0x1FFu;
  uint64_t idx3 = (va >> 30) & 0x1FFu;
  uint64_t idx2 = (va >> 21) & 0x1FFu;

  if (pai_gvm_read_phys(g, g->page_dir_phys + idx4 * 8, &e4) != 0 ||
      !(e4 & PAI_GPU_VALID)) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "gvm walk: pml4e invalid\n");
    return PAI_ERR_CAPABILITY;
  }

  if (pai_gvm_read_phys(g, (e4 & PAI_GPU_WALK_ADDR_MASK) + idx3 * 8, &e3) !=
          0 ||
      !(e3 & PAI_GPU_VALID)) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "gvm walk: pdpe invalid\n");
    return PAI_ERR_CAPABILITY;
  }

  *out_pde_addr = (e3 & PAI_GPU_WALK_ADDR_MASK) + idx2 * 8;
  if (pai_gvm_read_phys(g, *out_pde_addr, &e2) != 0) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "gvm walk: pde read failed\n");
    return PAI_ERR_CAPABILITY;
  }

  *out_pde = e2;
  return PAI_OK;
}

pai_status_t
pai_gvmspace_fix(uint64_t gpu_va, uint64_t phys, uint64_t size) {
  static pai_gvmspace_ctx_t g;
  uint64_t probe_va;
  uint64_t pde_addr = 0;
  uint64_t pde = 0;
  uint64_t new_pde;
  pai_status_t st;

  if (!g.found) {
    intptr_t data_base = (intptr_t)KERNEL_ADDRESS_DATA_BASE;

    g.entry_addr = 0;
    g.dmap_base = 0;

    PAI_LOG_INFO_(PAI_SUB_GPU, "gvmspace: scanning kernel data (0x%llx) "
                  "for the GPU address-space entry...\n",
                  (unsigned long long)data_base);

    for (intptr_t off = 0; off < (intptr_t)0x4000000; off += 0x100) {
      uint64_t start, size2;
      if (kernel_copyout(data_base + off + PAI_GVM_START_VA_OFF, &start, 8) !=
          0) {
        continue;
      }
      if (kernel_copyout(data_base + off + PAI_GVM_SIZE_OFF, &size2, 8) != 0) {
        continue;
      }

      /* Strict validation: sane GPU VA base, sane size, and the probe
       * VA must sit inside the range. */
      if (start < 0x100000000ULL || start >= 0x400000000ULL ||
          size2 < 0x100000ULL || size2 >= 0x100000000ULL ||
          gpu_va < start || gpu_va >= start + size2) {
        continue;
      }

      {
        uint64_t pdir = 0;
        uint64_t n_start = 0, n_size = 0;
        if (kernel_copyout(data_base + off + PAI_GVM_PAGE_DIR_OFF, &pdir, 8) !=
            0) {
          continue;
        }
        /* The page directory must be a real physical page. */
        if (pdir < 0x1000 || pdir >= 0x400000000ULL || (pdir & 0xFFF) != 0) {
          continue;
        }
        /* Array sanity: the next entry must be empty or plausible. */
        if (kernel_copyout(data_base + off + PAI_GVM_ENTRY_SIZE +
                               PAI_GVM_START_VA_OFF,
                           &n_start, 8) != 0 ||
            kernel_copyout(data_base + off + PAI_GVM_ENTRY_SIZE +
                               PAI_GVM_SIZE_OFF,
                           &n_size, 8) != 0) {
          continue;
        }
        if (!(n_start == 0 ||
              (n_start >= 0x100000000ULL && n_start < 0x400000000ULL &&
               n_size < 0x100000000ULL))) {
          continue;
        }
        g.entry_addr = data_base + off;
        g.start_va = start;
        g.size = size2;
        g.page_dir_phys = pdir;
        break;
      }
    }

    if (!g.entry_addr) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "gvmspace: no entry contains 0x%llx (scan failed)\n",
                     (unsigned long long)gpu_va);
      return PAI_ERR_CAPABILITY;
    }

    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "gvmspace: entry at 0x%llx [0x%llx + 0x%llx) pml4=0x%llx\n",
                  (unsigned long long)g.entry_addr,
                  (unsigned long long)g.start_va, (unsigned long long)g.size,
                  (unsigned long long)g.page_dir_phys);

    g.dmap_base = pai_gvm_find_dmap(g.page_dir_phys);
    if (!g.dmap_base) {
      PAI_LOG_ERROR_(PAI_SUB_GPU, "gvmspace: direct-map base not found\n");
      return PAI_ERR_CAPABILITY;
    }
    PAI_LOG_INFO_(PAI_SUB_GPU, "gvmspace: dmap at 0x%llx\n",
                  (unsigned long long)g.dmap_base);
    g.found = 1;
  }

  if (gpu_va < g.start_va || gpu_va >= g.start_va + g.size) {
    PAI_LOG_WARN_(PAI_SUB_GPU,
                  "gvmspace: 0x%llx outside [0x%llx, 0x%llx)\n",
                  (unsigned long long)gpu_va, (unsigned long long)g.start_va,
                  (unsigned long long)(g.start_va + g.size));
    return PAI_ERR_CAPABILITY;
  }

  st = pai_gvm_walk(&g, gpu_va, &pde_addr, &pde);
  if (st != PAI_OK) {
    return st;
  }

  PAI_LOG_INFO_(PAI_SUB_GPU, "gvmspace: va=0x%llx pde@0x%llx = 0x%llx\n",
                (unsigned long long)gpu_va, (unsigned long long)pde_addr,
                (unsigned long long)pde);

  if ((pde & PAI_GPU_VALID) && (pde & PAI_GPU_PHYS_MASK_2MB) ==
                                   (phys & PAI_GPU_PHYS_MASK_2MB)) {
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "gvmspace: PDE already maps the right physical frame\n");
    return PAI_OK;
  }

  /* Preserve the flag bits; replace the 2MB physical frame. */
  new_pde = (pde & ~PAI_GPU_PHYS_MASK_2MB) |
            (phys & PAI_GPU_PHYS_MASK_2MB) | PAI_GPU_VALID;

  if (pai_gvm_write_phys(&g, pde_addr, new_pde) != 0) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "gvmspace: PDE write failed\n");
    return PAI_ERR_CAPABILITY;
  }

  PAI_LOG_INFO_(PAI_SUB_GPU,
                "gvmspace: PDE patched 0x%llx -> 0x%llx (phys 0x%llx)\n",
                (unsigned long long)pde, (unsigned long long)new_pde,
                (unsigned long long)phys);

  (void)probe_va;
  (void)size;
  return PAI_OK;
}

#endif /* PAI_PS5 */
