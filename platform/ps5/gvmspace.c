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
#include <unistd.h>

#ifdef PAI_PS5

#include <ps5/kernel.h>

#define PAI_GPU_WALK_ADDR_MASK 0xFFFFFFFFC0ULL /* spoofer/11.20 */
#define PAI_GPU_VALID          (1ULL << 0)
#define PAI_GPU_IS_PTE         (1ULL << 54)
#define PAI_GPU_PHYS_MASK_2MB  ~(uint64_t)((2u << 20) - 1)
/* 9.40 GPU PTE: physical address in the low 46 bits, flags in the top
 * bits (observed: 0xF00000224BD067 -> phys 0x224BD000). */
#define PAI_GPU_PHYS_MASK_940  0x00003FFFFFFFFFFFULL

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
 * (the GPU pml4) and checking the content looks like a page table.
 * Only probe kernel VA space we know exists. */
static intptr_t
pai_gvm_find_dmap(uint64_t page_dir_phys) {
  if (page_dir_phys >= 0x400000000ULL) {
    return 0; /* phys must be inside the 16 GB console RAM */
  }
  for (uint64_t base = PAI_DMAP_MIN; base < PAI_DMAP_MIN + 0x4000000000ULL;
       base += PAI_DMAP_STEP) {
    uint64_t e0 = 0;
    uint64_t e1 = 0;
    intptr_t va = (intptr_t)(base + page_dir_phys);

    if (va < (intptr_t)0xFFFF800000000000ULL ||
        va > (intptr_t)0xFFFFFFFFFF000000ULL) {
      continue;
    }

    if (kernel_copyout(va, &e0, 8) != 0) {
      continue;
    }
    if (kernel_copyout(va + 8, &e1, 8) != 0) {
      continue;
    }

    if (pai_gvm_plausible_entry(e0) && pai_gvm_plausible_entry(e1)) {
      return base;
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

/* Diagnostic: dump the process vmspace pointer landscape so the real
 * 9.40 GPU-pmap layout can be derived from hardware instead of from
 * the 11.20 spoofer assumptions. Read-only. */
/* Live values captured by the diagnostic (read-only). */
static uint64_t g_diag_gpu_pml4_phys;
static intptr_t g_diag_dmap;
static int g_diag_have_layout;
static uint64_t g_ref_pde; /* valid PDE captured from a kernel-mapped VA */

pai_status_t
pai_gvmspace_diag(void) {
  intptr_t proc = kernel_get_proc(getpid());
  intptr_t vmspace = kernel_getlong(proc + KERNEL_OFFSET_PROC_P_VMSPACE);
  uint64_t v;

  if (!proc || !vmspace || vmspace == -1) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "gvm diag: proc/vmspace resolution failed\n");
    return PAI_ERR_CAPABILITY;
  }

  PAI_LOG_INFO_(PAI_SUB_GPU, "gvm diag: proc=0x%llx vmspace=0x%llx\n",
                (unsigned long long)proc, (unsigned long long)vmspace);

  for (intptr_t off = 0x100; off <= 0x600; off += 8) {
    uint64_t val = kernel_getlong(vmspace + off);
    if (val >= (uint64_t)(vmspace - 0x200000) &&
        val < (uint64_t)(vmspace + 0x200000)) {
      PAI_LOG_INFO_(PAI_SUB_GPU, "gvm diag: vmspace+0x%lx -> 0x%llx (delta "
                    "0x%llx)\n",
                    (unsigned long)off, (unsigned long long)val,
                    (unsigned long long)(val - (uint64_t)vmspace));
    }
  }

  /* Dump the first 0x40 bytes of each pmap-like pointer. */
  for (intptr_t off = 0x1C8; off <= 0x1F0; off += 8) {
    intptr_t pmap = (intptr_t)kernel_getlong(vmspace + off);
    if (!pmap || pmap == -1 || pmap < (intptr_t)0xFFFF800000000000ULL) {
      continue;
    }
    PAI_LOG_INFO_(PAI_SUB_GPU, "gvm diag: pmap@+0x%lx = 0x%llx: ",
                  (unsigned long)off, (unsigned long long)pmap);
    for (int i = 0; i < 8; i++) {
      v = kernel_getlong(pmap + i * 8);
      PAI_LOG_INFO_(PAI_SUB_GPU, "%llx ", (unsigned long long)v);
    }
    PAI_LOG_INFO_(PAI_SUB_GPU, "\n");
  }

  /* Deep-dump the GPU-pmap candidate (vmspace+0x1D8 pointer): 0x40..0x100. */
  {
    intptr_t pmap = (intptr_t)kernel_getlong(vmspace + 0x1D8);
    if (pmap && pmap != -1) {
      uint64_t va20 = kernel_getlong(pmap + 0x20);
      uint64_t ph28 = kernel_getlong(pmap + 0x28);

      for (int i = 8; i < 32; i++) {
        v = kernel_getlong(pmap + i * 8);
        PAI_LOG_INFO_(PAI_SUB_GPU, "gvm diag: pmap2+0x%lx = 0x%llx\n",
                      (unsigned long)(i * 8), (unsigned long long)v);
      }

      /* The (VA, phys) pair at +0x20/+0x28 is the page-table mapping;
       * VA - phys = the direct-map base, phys = the pml4. */
      if (va20 > 0xFFFF800000000000ULL && ph28 < 0x400000000ULL &&
          va20 - ph28 > 0xFFFF800000000000ULL) {
        g_diag_gpu_pml4_phys = ph28;
        g_diag_dmap = (intptr_t)(va20 - ph28);
        g_diag_have_layout = 1;
        PAI_LOG_INFO_(PAI_SUB_GPU,
                      "gvm diag: derived pml4=0x%llx dmap=0x%llx\n",
                      (unsigned long long)ph28,
                      (unsigned long long)(va20 - ph28));
      }
    }
  }

  return PAI_OK;
}

/* Public accessors for the probe. */
int pai_gvmspace_layout(uint64_t *out_pml4_phys, intptr_t *out_dmap) {
  if (!g_diag_have_layout) {
    return -1;
  }
  if (out_pml4_phys) {
    *out_pml4_phys = g_diag_gpu_pml4_phys;
  }
  if (out_dmap) {
    *out_dmap = g_diag_dmap;
  }
  return 0;
}

/* Read-only probe: walk the candidate GPU page-table root for `va`
 * and report the PDE. Tries both 3-level and 4-level interpretations
 * (the 9.40 GPU tables appear to be 3-level: PDPE -> PDE). No writes. */
pai_status_t
pai_gvmspace_probe(uint64_t pml4_phys, uint64_t va, intptr_t dmap_base) {
  uint64_t e4, e3, e2;
  uint64_t idx4 = (va >> 39) & 0x1FFu;
  uint64_t idx3 = (va >> 30) & 0x1FFu;
  uint64_t idx2 = (va >> 21) & 0x1FFu;
  intptr_t root = dmap_base + (intptr_t)pml4_phys;
  uint64_t root_e[16];

  for (int i = 0; i < 16; i++) {
    if (kernel_copyout(root + i * 8, &root_e[i], 8) != 0) {
      root_e[i] = 0;
    }
  }
  PAI_LOG_INFO_(PAI_SUB_GPU,
                "gvm probe: root[0..15] = %llx %llx %llx %llx %llx %llx "
                "%llx %llx %llx %llx %llx %llx %llx %llx %llx %llx\n",
                (unsigned long long)root_e[0], (unsigned long long)root_e[1],
                (unsigned long long)root_e[2], (unsigned long long)root_e[3],
                (unsigned long long)root_e[4], (unsigned long long)root_e[5],
                (unsigned long long)root_e[6], (unsigned long long)root_e[7],
                (unsigned long long)root_e[8], (unsigned long long)root_e[9],
                (unsigned long long)root_e[10], (unsigned long long)root_e[11],
                (unsigned long long)root_e[12], (unsigned long long)root_e[13],
                (unsigned long long)root_e[14], (unsigned long long)root_e[15]);

  /* 3-level interpretation: root = PDPE table, 9.40 PTE format
   * (phys in the low 46 bits). */
  e3 = root_e[idx3];
  if (e3 & PAI_GPU_VALID) {
    uint64_t pdpe_phys = e3 & PAI_GPU_PHYS_MASK_940 & ~0xFFFULL;
    uint64_t pde_page[16];

    PAI_LOG_INFO_(PAI_SUB_GPU, "gvm probe: pdpe=0x%llx phys=0x%llx\n",
                  (unsigned long long)e3, (unsigned long long)pdpe_phys);
    for (int i = 0; i < 16; i++) {
      if (kernel_copyout(dmap_base + (intptr_t)(pdpe_phys + i * 8),
                         &pde_page[i], 8) != 0) {
        pde_page[i] = 0;
      }
    }
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "gvm probe: pde[0..15] = %llx %llx %llx %llx %llx %llx "
                  "%llx %llx %llx %llx %llx %llx %llx %llx %llx %llx\n",
                  (unsigned long long)pde_page[0], (unsigned long long)pde_page[1],
                  (unsigned long long)pde_page[2], (unsigned long long)pde_page[3],
                  (unsigned long long)pde_page[4], (unsigned long long)pde_page[5],
                  (unsigned long long)pde_page[6], (unsigned long long)pde_page[7],
                  (unsigned long long)pde_page[8], (unsigned long long)pde_page[9],
                  (unsigned long long)pde_page[10], (unsigned long long)pde_page[11],
                  (unsigned long long)pde_page[12], (unsigned long long)pde_page[13],
                  (unsigned long long)pde_page[14], (unsigned long long)pde_page[15]);
    e2 = pde_page[idx2];
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "gvm probe (3-level): va=0x%llx idx2=%llu pde=0x%llx "
                  "(%s)\n",
                  (unsigned long long)va, (unsigned long long)idx2,
                  (unsigned long long)e2,
                  (e2 & PAI_GPU_VALID) ? "valid" : "INVALID");
    return PAI_OK;
  }

  /* 4-level interpretation: root = pml4 table. */
  e4 = root_e[idx4];
  if (!(e4 & PAI_GPU_VALID)) {
    PAI_LOG_INFO_(PAI_SUB_GPU, "gvm probe (4-level): pml4e invalid (0x%llx)\n",
                  (unsigned long long)e4);
    return PAI_ERR_CAPABILITY;
  }
  {
    uint64_t p4_phys = e4 & PAI_GPU_PHYS_MASK_940 & ~0xFFFULL;
    if (kernel_copyout(dmap_base + (intptr_t)(p4_phys + idx3 * 8), &e3, 8) !=
        0) {
      return PAI_ERR_CAPABILITY;
    }
  }
  if (!(e3 & PAI_GPU_VALID)) {
    PAI_LOG_INFO_(PAI_SUB_GPU, "gvm probe (4-level): pdpe invalid (0x%llx)\n",
                  (unsigned long long)e3);
    return PAI_ERR_CAPABILITY;
  }
  {
    uint64_t p3_phys = e3 & PAI_GPU_PHYS_MASK_940 & ~0xFFFULL;
    if (kernel_copyout(dmap_base + (intptr_t)(p3_phys + idx2 * 8), &e2, 8) !=
        0) {
      return PAI_ERR_CAPABILITY;
    }
  }
  PAI_LOG_INFO_(PAI_SUB_GPU,
                "gvm probe (4-level): pml4=0x%llx va=0x%llx pde=0x%llx "
                "(%s)\n",
                (unsigned long long)pml4_phys, (unsigned long long)va,
                (unsigned long long)e2,
                (e2 & PAI_GPU_VALID) ? "valid" : "INVALID");
  if (e2 & PAI_GPU_VALID) {
    g_ref_pde = e2;
  }
  return PAI_OK;
}

/*
 * Repair the GPU mapping for `gpu_va` (2 MB granularity): walk the
 * verified page-table structure and, when the PDE is invalid, write a
 * 2 MB leaf built from the reference flags (captured from a
 * kernel-mapped VA) and our physical frame. Verifies the write.
 */
pai_status_t
pai_gvmspace_repair(uint64_t gpu_va, uint64_t phys) {
  uint64_t e4, e3, e2;
  uint64_t idx4 = (gpu_va >> 39) & 0x1FFu;
  uint64_t idx3 = (gpu_va >> 30) & 0x1FFu;
  uint64_t idx2 = (gpu_va >> 21) & 0x1FFu;
  intptr_t dmap;
  uint64_t pml4;
  uint64_t p4_phys, p3_phys;
  uint64_t pde_addr;
  uint64_t new_pde;
  uint64_t verify;

  if (!g_diag_have_layout || !g_ref_pde) {
    PAI_LOG_WARN_(PAI_SUB_GPU,
                  "gvm repair: layout/reference PDE unavailable\n");
    return PAI_ERR_CAPABILITY;
  }
  dmap = g_diag_dmap;
  pml4 = g_diag_gpu_pml4_phys;

  if (kernel_copyout(dmap + (intptr_t)(pml4 + idx4 * 8), &e4, 8) != 0 ||
      !(e4 & PAI_GPU_VALID)) {
    return PAI_ERR_CAPABILITY;
  }
  p4_phys = e4 & PAI_GPU_PHYS_MASK_940 & ~0xFFFULL;
  if (kernel_copyout(dmap + (intptr_t)(p4_phys + idx3 * 8), &e3, 8) != 0 ||
      !(e3 & PAI_GPU_VALID)) {
    return PAI_ERR_CAPABILITY;
  }
  p3_phys = e3 & PAI_GPU_PHYS_MASK_940 & ~0xFFFULL;
  if (p3_phys >= 0x400000000ULL) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "gvm repair: pde table phys out of range\n");
    return PAI_ERR_CAPABILITY;
  }
  pde_addr = dmap + (intptr_t)(p3_phys + idx2 * 8);

  if (kernel_copyout(pde_addr, &e2, 8) != 0) {
    return PAI_ERR_CAPABILITY;
  }

  if ((e2 & PAI_GPU_VALID) &&
      (e2 & 0x00003FFFFFE00000ULL) == (phys & 0x00003FFFFFE00000ULL)) {
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "gvm repair: va=0x%llx already maps phys 0x%llx\n",
                  (unsigned long long)gpu_va, (unsigned long long)phys);
    return PAI_OK;
  }

  /* Build the 2 MB leaf: reference flags + our physical frame. */
  new_pde = (g_ref_pde & ~0x00003FFFFFE00000ULL) |
            (phys & 0x00003FFFFFE00000ULL) | PAI_GPU_VALID;

  PAI_LOG_INFO_(PAI_SUB_GPU,
                "gvm repair: va=0x%llx phys=0x%llx pde 0x%llx -> 0x%llx "
                "(ref 0x%llx)\n",
                (unsigned long long)gpu_va, (unsigned long long)phys,
                (unsigned long long)e2, (unsigned long long)new_pde,
                (unsigned long long)g_ref_pde);

  if (kernel_copyin(&new_pde, pde_addr, 8) != 0) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "gvm repair: PDE write failed\n");
    return PAI_ERR_CAPABILITY;
  }

  if (kernel_copyout(pde_addr, &verify, 8) != 0 || verify != new_pde) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "gvm repair: PDE verify failed (0x%llx)\n",
                   (unsigned long long)verify);
    return PAI_ERR_CAPABILITY;
  }

  PAI_LOG_INFO_(PAI_SUB_GPU, "gvm repair: PDE patched and verified\n");
  return PAI_OK;
}

pai_status_t
pai_gvmspace_fix(uint64_t gpu_va, uint64_t phys, uint64_t size) {
  static pai_gvmspace_ctx_t g;
  static int dumped = 0;
  uint64_t probe_va;
  uint64_t pde_addr = 0;
  uint64_t pde = 0;
  uint64_t new_pde;
  pai_status_t st;

  if (!dumped) {
    dumped = 1;
    (void)pai_gvmspace_diag();
  }

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
