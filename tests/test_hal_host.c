#include "test.h"

#include <hal/hal.h>
#include <host_kernels.h>
#include <m0_experiments.h>
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

  /* M0 kernels: store_const / loadstore through the shared code buffer
   * (exercises shader re-registration) + FLAT bit-15 patch targets. */
  {
    pai_pm4_builder_t pb;
    uint32_t pm4[64];
    uint32_t vals[4];

    /* The bit-15 patch must hit FLAT word0 (DC prefix, bit 15 clear).
     * word1 carries ADDR/DATA — patching it corrupts a register field
     * or an immediate. */
    CHECK((pai_store_const_code[PAI_STORE_CONST_FLAT_WORD] &
           0xFF000000u) == 0xDC000000u);
    CHECK(!(pai_store_const_code[PAI_STORE_CONST_FLAT_WORD] & 0x8000u));
    CHECK((pai_loadstore_code[PAI_LOADSTORE_FLAT_LOAD_WORD] &
           0xFF000000u) == 0xDC000000u);
    CHECK(!(pai_loadstore_code[PAI_LOADSTORE_FLAT_LOAD_WORD] & 0x8000u));
    CHECK((pai_loadstore_code[PAI_LOADSTORE_FLAT_STORE_WORD] &
           0xFF000000u) == 0xDC000000u);
    CHECK(!(pai_loadstore_code[PAI_LOADSTORE_FLAT_STORE_WORD] & 0x8000u));
    CHECK_EQ_UINT(pai_store_const_code[PAI_STORE_CONST_FLAT_WORD],
                  0xDC700000u);
    CHECK_EQ_UINT(pai_loadstore_code[PAI_LOADSTORE_FLAT_LOAD_WORD],
                  0xDC300000u);
    CHECK_EQ_UINT(pai_loadstore_code[PAI_LOADSTORE_FLAT_STORE_WORD],
                  0xDC700000u);

    /* store_const (replaces the vecadd registration at the same addr) */
    memcpy(code.cpu_addr, pai_store_const_code,
           PAI_STORE_CONST_CODE_WORDS * sizeof(uint32_t));
    CHECK(pai_gpu_host_register_shader(dev, code.gpu_addr,
                                       pai_host_kernel_store_const, NULL) ==
          PAI_OK);
    memset(c.cpu_addr, 0, 4096);

    pai_pm4_builder_init(&pb, pm4, 64);
    vals[0] = (uint32_t)(code.gpu_addr >> 8);
    vals[1] = (uint32_t)(code.gpu_addr >> 40);
    pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_LO, 2, vals);
    vals[0] = PAI_EXP_RSRC1;
    vals[1] = PAI_STORE_CONST_RSRC2;
    pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC1, 2, vals);
    vals[0] = PAI_EXP_RSRC3;
    pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC3, 1, vals);
    vals[0] = PAI_EXP_THREADS_X;
    vals[1] = 1;
    vals[2] = 1;
    pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_NUM_THREAD_X, 3, vals);
    vals[0] = (uint32_t)(c.gpu_addr & 0xFFFFFFFFu);
    vals[1] = (uint32_t)(c.gpu_addr >> 32);
    pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_USER_DATA_0, 2, vals);
    pai_pm4_dispatch_direct(&pb, 1, 1, 1, 0);

    *(volatile uint32_t *)label.cpu_addr = 0;
    CHECK(pai_gpu_submit_wait(dev, pm4, pb.len, label.gpu_addr, 0x55,
                              UINT64_C(1000000000)) == PAI_OK);
    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      CHECK_EQ_UINT(((uint32_t *)c.cpu_addr)[i], PAI_STORE_CONST_VALUE);
    }
    CHECK_EQ_UINT(((uint32_t *)c.cpu_addr)[PAI_EXP_THREADS_X], 0);

    /* loadstore (replaces the store_const registration at the same addr) */
    memcpy(code.cpu_addr, pai_loadstore_code,
           PAI_LOADSTORE_CODE_WORDS * sizeof(uint32_t));
    CHECK(pai_gpu_host_register_shader(dev, code.gpu_addr,
                                       pai_host_kernel_loadstore, NULL) ==
          PAI_OK);
    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      ((uint32_t *)a.cpu_addr)[i] = 0x11111111u + i;
    }
    memset(c.cpu_addr, 0xCC, 4096);

    pai_pm4_builder_init(&pb, pm4, 64);
    vals[0] = (uint32_t)(code.gpu_addr >> 8);
    vals[1] = (uint32_t)(code.gpu_addr >> 40);
    pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_LO, 2, vals);
    vals[0] = PAI_EXP_RSRC1;
    vals[1] = PAI_LOADSTORE_RSRC2;
    pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC1, 2, vals);
    vals[0] = PAI_EXP_RSRC3;
    pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC3, 1, vals);
    vals[0] = PAI_EXP_THREADS_X;
    vals[1] = 1;
    vals[2] = 1;
    pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_NUM_THREAD_X, 3, vals);
    vals[0] = (uint32_t)(a.gpu_addr & 0xFFFFFFFFu);
    vals[1] = (uint32_t)(a.gpu_addr >> 32);
    vals[2] = (uint32_t)(c.gpu_addr & 0xFFFFFFFFu);
    vals[3] = (uint32_t)(c.gpu_addr >> 32);
    pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_USER_DATA_0, 4, vals);
    pai_pm4_dispatch_direct(&pb, 1, 1, 1, 0);

    *(volatile uint32_t *)label.cpu_addr = 0;
    CHECK(pai_gpu_submit_wait(dev, pm4, pb.len, label.gpu_addr, 0x56,
                              UINT64_C(1000000000)) == PAI_OK);
    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      CHECK_EQ_UINT(((uint32_t *)c.cpu_addr)[i], 0x11111111u + i);
    }
    CHECK_EQ_UINT(((uint32_t *)c.cpu_addr)[PAI_EXP_THREADS_X], 0xCCCCCCCCu);
  }

  /* unknown opcode must fail loud, not silently */
  {
    uint32_t bad[2] = {0xC000FF00u, 0};
    CHECK(pai_gpu_submit_wait(dev, bad, 2, label.gpu_addr, 1,
                              UINT64_C(1000000000)) == PAI_ERR_UNSUPPORTED);
  }

  /* T4B integer kernels (int_ops.s G49-G54): host-ref mirrors must
   * match the u32 wrap semantics of the serial shaders one-to-one. */
  {
    uint32_t ab[2 * 64];
    uint32_t cbuf[64];
    uint32_t ud[7] = {0};
    uint32_t want;

    for (uint32_t i = 0; i < 64; i++) {
      ab[2 * i] = 0x80000000u + 0x01000000u * i;
      ab[2 * i + 1] = 0x11111111u + 0x00001000u * i;
    }
    ud[2] = (uint32_t)(uintptr_t)ab;
    ud[3] = (uint32_t)((uintptr_t)ab >> 32);
    ud[4] = (uint32_t)(uintptr_t)cbuf;
    ud[5] = (uint32_t)((uintptr_t)cbuf >> 32);

    /* add2d */
    CHECK(pai_host_kernel_int_add2d(NULL, ud, 1, 64) == PAI_OK);
    for (uint32_t i = 0; i < 64; i++) {
      CHECK_EQ_UINT(cbuf[i], ab[2 * i] + ab[2 * i + 1]);
    }
    /* sub1d */
    CHECK(pai_host_kernel_int_sub1d(NULL, ud, 1, 64) == PAI_OK);
    for (uint32_t i = 0; i < 64; i++) {
      CHECK_EQ_UINT(cbuf[i], ab[2 * i] - ab[2 * i + 1]);
    }
    /* mul1d */
    CHECK(pai_host_kernel_int_mul1d(NULL, ud, 1, 64) == PAI_OK);
    for (uint32_t i = 0; i < 64; i++) {
      CHECK_EQ_UINT(cbuf[i], ab[2 * i] * ab[2 * i + 1]);
    }
    /* relu */
    CHECK(pai_host_kernel_int_relu(NULL, ud, 1, 64) == PAI_OK);
    for (uint32_t i = 0; i < 64; i++) {
      CHECK_EQ_UINT(cbuf[i], ab[2 * i] > 0u ? ab[2 * i] : 0u);
    }
    /* clip */
    CHECK(pai_host_kernel_int_clip(NULL, ud, 1, 64) == PAI_OK);
    for (uint32_t i = 0; i < 64; i++) {
      CHECK_EQ_UINT(cbuf[i], ab[2 * i] < 1u ? ab[2 * i] : 1u);
    }

    /* matmul_u32: header [K, N, a_lo, a_hi, b_lo, b_hi] */
    {
      uint32_t h[6];
      uint32_t a48[4 * 16];
      uint32_t b48[16 * 4];
      uint32_t rows = 4, kk = 16, cols = 4;

      for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t t = 0; t < kk; t++) {
          a48[i * kk + t] = 0x10000000u + 0x00000100u * i + t;
        }
      }
      for (uint32_t t = 0; t < kk; t++) {
        for (uint32_t j = 0; j < cols; j++) {
          b48[t * cols + j] = 0x20000000u + 0x00010000u * t + j;
        }
      }
      h[0] = kk;
      h[1] = cols;
      h[2] = (uint32_t)(uintptr_t)a48;
      h[3] = (uint32_t)((uintptr_t)a48 >> 32);
      h[4] = (uint32_t)(uintptr_t)b48;
      h[5] = (uint32_t)((uintptr_t)b48 >> 32);
      ud[2] = (uint32_t)(uintptr_t)h;
      ud[3] = (uint32_t)((uintptr_t)h >> 32);
      CHECK(pai_host_kernel_int_matmul(NULL, ud, 1, rows) == PAI_OK);
      for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
          want = 0;
          for (uint32_t t = 0; t < kk; t++) {
            want += a48[i * kk + t] * b48[t * cols + j];
          }
          CHECK_EQ_UINT(cbuf[i * cols + j], want);
        }
      }
    }
  }

  pai_gpu_buffer_free(dev, &label);
  pai_gpu_buffer_free(dev, &code);
  pai_gpu_buffer_free(dev, &c);
  pai_gpu_buffer_free(dev, &b);
  pai_gpu_buffer_free(dev, &a);
  pai_gpu_device_destroy(dev);
}

TEST_MAIN_END()
