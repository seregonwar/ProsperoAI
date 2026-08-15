#include "test.h"

#include <hal/hal.h>
#include <host_kernels.h>
#include <pm4/pm4.h>
#include <ref_ops.h>
#include <vecadd.h>

#include <stdlib.h>
#include <string.h>

/*
 * End-to-end test of the host reference GPU backend: the same PM4
 * streams the PS5 backend submits (DMA_DATA, compute dispatch, EOP
 * fences) executed by the interpreter.
 */

static uint32_t
build_vecadd_stream(uint32_t *pm4, uint32_t cap, uint64_t a_addr,
                    uint64_t b_addr, uint64_t c_addr, uint64_t code_addr,
                    uint32_t n) {
  pai_pm4_builder_t b;
  uint32_t vals[8];

  pai_pm4_builder_init(&b, pm4, cap);

  vals[0] = (uint32_t)(code_addr >> 8);
  vals[1] = (uint32_t)(code_addr >> 40);
  pai_pm4_set_sh_reg_compute(&b, PAI_REG_COMPUTE_PGM_LO, 2, vals);

  vals[0] = PAI_VECADD_RSRC1;
  vals[1] = PAI_VECADD_RSRC2;
  pai_pm4_set_sh_reg_compute(&b, PAI_REG_COMPUTE_PGM_RSRC1, 2, vals);

  vals[0] = PAI_VECADD_RSRC3;
  pai_pm4_set_sh_reg_compute(&b, PAI_REG_COMPUTE_PGM_RSRC3, 1, vals);

  vals[0] = PAI_VECADD_THREADS_X;
  vals[1] = 1;
  vals[2] = 1;
  pai_pm4_set_sh_reg_compute(&b, PAI_REG_COMPUTE_NUM_THREAD_X, 3, vals);

  vals[0] = (uint32_t)(a_addr & 0xFFFFFFFFu);
  vals[1] = (uint32_t)(a_addr >> 32);
  vals[2] = (uint32_t)(b_addr & 0xFFFFFFFFu);
  vals[3] = (uint32_t)(b_addr >> 32);
  vals[4] = (uint32_t)(c_addr & 0xFFFFFFFFu);
  vals[5] = (uint32_t)(c_addr >> 32);
  vals[6] = n;
  pai_pm4_set_sh_reg_compute(&b, PAI_REG_COMPUTE_USER_DATA_0, 7, vals);

  pai_pm4_dispatch_direct(&b, 1, 1, 1, 0);

  return b.len;
}

TEST_MAIN_BEGIN()

{
  pai_gpu_device_t *dev = NULL;
  pai_gpu_buffer_t a, b, c, code, label;
  uint32_t pm4[64];
  uint32_t pm4_len;
  uint32_t n = PAI_VECADD_MAX_ELEMS;
  float ref[PAI_VECADD_MAX_ELEMS];
  uint64_t mismatch = 0;

  CHECK(pai_gpu_device_create(PAI_GPU_BACKEND_HOST_REF, &dev) == PAI_OK);
  if (!dev) {
    return 1;
  }

  CHECK(pai_gpu_buffer_alloc(dev, &a, 4096, PAI_GPU_BUF_CPU_VISIBLE) ==
        PAI_OK);
  CHECK(pai_gpu_buffer_alloc(dev, &b, 4096, PAI_GPU_BUF_CPU_VISIBLE) ==
        PAI_OK);
  CHECK(pai_gpu_buffer_alloc(dev, &c, 4096, PAI_GPU_BUF_CPU_VISIBLE) ==
        PAI_OK);
  CHECK(pai_gpu_buffer_alloc(dev, &code, 4096, PAI_GPU_BUF_CPU_VISIBLE) ==
        PAI_OK);
  CHECK(pai_gpu_buffer_alloc(dev, &label, 4096, PAI_GPU_BUF_CPU_VISIBLE) ==
        PAI_OK);

  for (uint32_t i = 0; i < n; i++) {
    ((float *)a.cpu_addr)[i] = (float)i * 0.25f;
    ((float *)b.cpu_addr)[i] = (float)(i % 3) + 0.5f;
  }
  memset(c.cpu_addr, 0xCC, 4096);

  memcpy(code.cpu_addr, pai_vecadd_code,
         PAI_VECADD_CODE_WORDS * sizeof(uint32_t));

  CHECK(pai_gpu_host_register_shader(dev, code.gpu_addr,
                                     pai_host_kernel_vecadd, NULL) == PAI_OK);

  pm4_len = build_vecadd_stream(pm4, 64, a.gpu_addr, b.gpu_addr, c.gpu_addr,
                                code.gpu_addr, n);

  *(volatile uint32_t *)label.cpu_addr = 0;
  CHECK(pai_gpu_submit_wait(dev, pm4, pm4_len, label.gpu_addr, 0x1234,
                            UINT64_C(1000000000)) == PAI_OK);
  CHECK_EQ_UINT(*(volatile uint32_t *)label.cpu_addr, 0x1234);

  CHECK(pai_ref_vecadd_f32((float *)a.cpu_addr, (float *)b.cpu_addr, ref,
                           n) == PAI_OK);
  CHECK(pai_ref_compare_f32((float *)c.cpu_addr, ref, n, 1e-6f, 1e-6f,
                            &mismatch) == PAI_OK);

  /* DMA_DATA path */
  {
    pai_pm4_builder_t db;
    uint32_t dma[16];
    pai_pm4_builder_init(&db, dma, 16);
    pai_pm4_dma_data(&db, a.gpu_addr, c.gpu_addr, 1024);
    memset(c.cpu_addr, 0, 1024);
    *(volatile uint32_t *)label.cpu_addr = 0;
    CHECK(pai_gpu_submit_wait(dev, dma, db.len, label.gpu_addr, 0x99,
                              UINT64_C(1000000000)) == PAI_OK);
    CHECK(memcmp(a.cpu_addr, c.cpu_addr, 1024) == 0);
  }

  /* unknown opcode must fail loud, not silently */
  {
    uint32_t bad[2] = {0xC000FF00u, 0};
    CHECK(pai_gpu_submit_wait(dev, bad, 2, label.gpu_addr, 1,
                              UINT64_C(1000000000)) == PAI_ERR_UNSUPPORTED);
  }

  pai_gpu_buffer_free(dev, &label);
  pai_gpu_buffer_free(dev, &code);
  pai_gpu_buffer_free(dev, &c);
  pai_gpu_buffer_free(dev, &b);
  pai_gpu_buffer_free(dev, &a);
  pai_gpu_device_destroy(dev);
}

TEST_MAIN_END()
