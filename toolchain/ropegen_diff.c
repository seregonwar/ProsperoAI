/*
 * ProsperoAI — Phase 2 differential: on-GPU RoPE table generator ABI
 * (G67/G68 ropegen.s) vs the double oracle.
 *
 * A's ropegen kernel (HW-validated on console 9021) produces the
 * cos_t/sin_t tables the decoder consumes. This harness is Seat B's
 * leg: it drives the host mirror of the exact kernel ABI — header
 * (r2, ctx, theta_turns[0..ctx*r2-1]) at ud[2:3], C at ud[4:5], one
 * group per element, cos entry writes c[e], sin entry writes the
 * ctx*r2 half — and cross-checks the result against the independent
 * pai_ref_rope_cossin_f32 oracle (double) plus the cos^2+sin^2=1
 * identity.
 *
 * Contract (fixed in tests/test_ref_ops.c, commit 3f25b86):
 *   inv_freq[i] = base^(-i/r2)
 *   theta[p][i] = p * inv_freq[i]
 *   theta_turns[e] = theta[p][i] / (2*pi), e = p*r2+i
 *   c[e] = cos/sin(2*pi*theta_turns[e]) == cos/sin(theta[p][i])
 *
 *   sin half (9.40): v_sin_f32 is TOXIC serial-per-element (commit
 *   ca2819e) — the payload feeds theta_turns-0.25 and the cos kernel
 *   produces sin(theta) via cos(theta-pi/2). Section 5 locks that.
 *
 * Build: host-tests/host-reference preset; binary `ropegen_diff`.
 */

#include <hal/host_kernels.h>
#include <ref_ops.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RG_CTX  32u
#define RG_R2   4u
#define RG_BASE 10000.0f
#define RG_TWO_PI 6.28318530717958647692

static int g_failures;

#define REQUIRE(cond)                                                        \
  do {                                                                       \
    if (!(cond)) {                                                           \
      printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);               \
      g_failures++;                                                          \
    }                                                                        \
  } while (0)

int
main(void) {
  const uint32_t ctx = RG_CTX, r2 = RG_R2, n = ctx * r2;
  uint32_t hdr[2 + 32 * 4];   /* r2, ctx, theta_turns[n] */
  uint32_t ud[16] = {0};
  float tables[2 * 32 * 4];   /* cos at [0..n), sin at [n..2n) */
  float cos_o[32 * 4], sin_o[32 * 4];
  uint64_t mism = 0;
  pai_status_t st;

  printf("== Phase 2 differential: ropegen G67/G68 ABI vs oracle "
         "(ctx=%u r2=%u base=%.0f) ==\n", ctx, r2, (double)RG_BASE);

  /* theta_turns per contract: theta[p][i] = p*base^(-i/r2), /2pi. */
  hdr[0] = r2;
  hdr[1] = ctx;
  for (uint32_t p = 0; p < ctx; p++) {
    for (uint32_t i = 0; i < r2; i++) {
      double inv_freq = pow((double)RG_BASE, -(double)i / (double)r2);
      double theta = (double)p * inv_freq;
      float tt = (float)(theta / RG_TWO_PI);
      memcpy(&hdr[2 + p * r2 + i], &tt, 4);
    }
  }

  /* Oracle: pai_ref_rope_cossin_f32 (the decoder's own table source). */
  st = pai_ref_rope_cossin_f32(ctx, r2, RG_BASE, cos_o, sin_o);
  REQUIRE(st == PAI_OK);

  /* Drive the G67/G68 mirror ABI exactly as the payload dispatches it:
   * header at ud[2:3], C at ud[4:5], one group per element, cos then
   * sin (sin writes the ctx*r2 half). */
  memset(tables, 0xCC, sizeof(tables));
  ud[2] = (uint32_t)(uintptr_t)hdr;
  ud[3] = (uint32_t)((uintptr_t)hdr >> 32);
  ud[4] = (uint32_t)(uintptr_t)tables;
  ud[5] = (uint32_t)((uintptr_t)tables >> 32);
  st = pai_host_kernel_ropegen_cos(NULL, ud, 1, n);
  REQUIRE(st == PAI_OK);
  st = pai_host_kernel_ropegen_sin(NULL, ud, 1, n);
  REQUIRE(st == PAI_OK);

  /* 1) cos half vs oracle (pai_ref_compare_f32: PAI_OK + sentinel
   * UINT64_MAX means no mismatch). */
  mism = 0;
  st = pai_ref_compare_f32(tables, cos_o, n, 1e-4f, 1e-4f, &mism);
  if (st != PAI_OK) {
    printf("  FAIL cos_t: first mismatch e=%llu (got %.7f want %.7f)\n",
           (unsigned long long)mism, mism < n ? (double)tables[mism] : 0.0,
           mism < n ? (double)cos_o[mism] : 0.0);
    g_failures++;
  } else {
    printf("  PASS cos_t[0..%u) == oracle\n", n);
  }

  /* 2) sin half vs oracle. */
  mism = 0;
  st = pai_ref_compare_f32(tables + n, sin_o, n, 1e-4f, 1e-4f, &mism);
  if (st != PAI_OK) {
    printf("  FAIL sin_t: first mismatch e=%llu (got %.7f want %.7f)\n",
           (unsigned long long)mism, mism < n ? (double)tables[n + mism] : 0.0,
           mism < n ? (double)sin_o[mism] : 0.0);
    g_failures++;
  } else {
    printf("  PASS sin_t[n..2n) == oracle\n");
  }

  /* 3) identity cos^2+sin^2 = 1 on every pair (HW check from G65). */
  {
    uint64_t worst = 0;
    double max_d = 0.0;
    for (uint32_t e = 0; e < n; e++) {
      double d = fabs((double)tables[e] * tables[e] +
                      (double)tables[n + e] * tables[n + e] - 1.0);
      if (d > max_d) {
        max_d = d;
        worst = e;
      }
    }
    if (max_d < 1e-5) {
      printf("  PASS cos^2+sin^2 == 1 (max |d| = %g)\n", max_d);
    } else {
      printf("  FAIL identity at e=%llu: max |d| = %g\n",
             (unsigned long long)worst, max_d);
      g_failures++;
    }
  }

  /* 4) consumability: feeding the ABI-produced tables into rope_f32
   * must reproduce the oracle-table result exactly (same rows). 3 rows
   * x hd=8 -> 24 elements. */
  {
    float x[3 * 8], y_abi[3 * 8], y_or[3 * 8];
    for (uint32_t i = 0; i < 3 * 8; i++) {
      x[i] = (float)(((i * 7 + 3) % 13) - 6) * 0.1f;
    }
    st = pai_ref_rope_f32(x, 3, 8, 3, 1, tables, tables + n, r2, y_abi);
    REQUIRE(st == PAI_OK);
    st = pai_ref_rope_f32(x, 3, 8, 3, 1, cos_o, sin_o, r2, y_or);
    REQUIRE(st == PAI_OK);
    mism = 0;
    st = pai_ref_compare_f32(y_abi, y_or, 24, 1e-4f, 1e-4f, &mism);
    if (st != PAI_OK) {
      printf("  FAIL rope_f32 with ABI tables != oracle tables "
             "(first %llu)\n", (unsigned long long)mism);
      g_failures++;
    } else {
      printf("  PASS rope_f32(ABI tables) == rope_f32(oracle tables)\n");
    }
  }

  /* 5) G70 cos-shift contract: on 9.40 v_sin_f32 is TOXIC in
   * serial-per-element (EOP never fires, commit ca2819e), so the sin
   * half is produced by feeding theta_turns-0.25 to the cos kernel
   * (cos(theta-pi/2)=sin(theta)). Validate the exact recipe host-side:
   * mirror sin with the shifted header must equal the oracle sin, and
   * the cos+shifted-sin pair keeps cos^2+sin^2==1. */
  {
    uint32_t hdr_shift[2 + 32 * 4];
    float tables2[2 * 32 * 4];
    uint32_t ud2[16] = {0};
    double max_d = 0.0, worst_id = 0.0;
    uint64_t worst_e = 0, worst_id_e = 0;

    hdr_shift[0] = r2;
    hdr_shift[1] = ctx;
    for (uint32_t e = 0; e < n; e++) {
      float tt;
      memcpy(&tt, &hdr[2 + e], 4);
      tt -= 0.25f; /* the G70 shift: cos(theta-pi/2) */
      memcpy(&hdr_shift[2 + e], &tt, 4);
    }
    memset(tables2, 0xCC, sizeof(tables2));
    ud2[2] = (uint32_t)(uintptr_t)hdr_shift;
    ud2[3] = (uint32_t)((uintptr_t)hdr_shift >> 32);
    ud2[4] = (uint32_t)(uintptr_t)tables2;
    ud2[5] = (uint32_t)((uintptr_t)tables2 >> 32);
    /* The G70 kernel is the COS instruction fed the shifted header
     * (G69 patched the v_sin word to v_cos in place); the host analog
     * is the cos mirror with tt-0.25. Values land in the cos half. */
    st = pai_host_kernel_ropegen_cos(NULL, ud2, 1, n);
    REQUIRE(st == PAI_OK);
    for (uint32_t e = 0; e < n; e++) {
      double d = fabs((double)tables2[e] - (double)sin_o[e]);
      if (d > max_d) {
        max_d = d;
        worst_e = e;
      }
      double id = fabs((double)tables[e] * tables[e] +
                       (double)tables2[e] * tables2[e] - 1.0);
      if (id > worst_id) {
        worst_id = id;
        worst_id_e = e;
      }
    }
    if (max_d < 1e-4 && worst_id < 1e-5) {
      printf("  PASS G70 cos-shift sin == oracle sin (max |d| = %g), "
             "cos^2+sin^2 = 1 (max |d| = %g)\n", max_d, worst_id);
    } else {
      printf("  FAIL G70 cos-shift: sin max |d| = %g (e=%llu), "
             "identity max |d| = %g (e=%llu)\n", max_d,
             (unsigned long long)worst_e, worst_id,
             (unsigned long long)worst_id_e);
      g_failures++;
    }
  }

  printf("== ropegen_diff %s ==\n", g_failures ? "FAILED" : "PASSED");
  return g_failures ? 1 : 0;
}
