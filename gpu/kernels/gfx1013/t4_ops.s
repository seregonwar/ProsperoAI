# t4_ops.s - PAI-M0 experiments G42-G48 (T4 serial float kernels)
#
# All kernels use the validated e64 form (G35/G39/G40/G41):
# direct-SGPR operands, dst != v0, v0 = value for the store.
# ABI: s2:s3 = packed inputs, s4:s5 = C, s6 = TGID_X, RSRC2 = 0x8C,
# groups_x = N (elementwise) or rows*cols (biasadd/matmul), 1 thread.
#
# Elementwise (G42-G46), packed (a,b) pairs at dword[2g], dword[2g+1]:
#   add1d: c[g] = a[g] + b[g]
#   sub1d: c[g] = a[g] - b[g]
#   mul1d: c[g] = a[g] * b[g]
#   relu:  c[g] = max(a[g], 0)
#   clip:  c[g] = clamp(a[g], 0, 1)
# Biasadd (G47): c[g] = a[g] + bias[j], header [cols, pad, a_lo, a_hi,
#   bias_lo, bias_hi], groups = rows*cols, j = g % cols.
# Matmul (G48): c[g] = sum_k a[i*K+k] * b[k*N+j], header [K, N,
#   a_lo, a_hi, b_lo, b_hi], groups = rows*N, i = g / N, j = g % N.
#
# HW rule (G47/G48 first runs): a flat_store_dword INSIDE a loop with
# an s_cbranch back edge hangs the wave after the first store. All
# loop bodies therefore accumulate without storing; each group does
# exactly ONE store at the end (G22/G40-validated pattern).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj t4_ops.s -o t4_ops.o
#   llvm-objcopy --dump-section .text=t4_ops.bin t4_ops.o

.text
.globl t4_add1d
.globl t4_sub1d
.globl t4_mul1d
.globl t4_relu
.globl t4_clip
.globl t4_biasadd
.globl t4_matmul

.macro t4_store
    s_lshl_b32 s16, s6, 2
    s_add_u32 s18, s4, s16
    s_addc_u32 s19, s5, 0
    v_mov_b32 v2, s18
    v_mov_b32 v3, s19
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm
.endm

.macro t4_load_pair
    s_lshl_b32 s16, s6, 3
    s_add_u32 s18, s2, s16
    s_addc_u32 s19, s3, 0
    s_load_dword s20, s[18:19], 0
    s_load_dword s21, s[18:19], 4
    s_waitcnt lgkmcnt(0)
.endm

t4_add1d:
    t4_load_pair
    v_add_f32_e64 v1, s20, s21
    t4_store

t4_sub1d:
    t4_load_pair
    v_sub_f32_e64 v1, s20, s21
    t4_store

t4_mul1d:
    t4_load_pair
    v_mul_f32_e64 v1, s20, s21
    t4_store

t4_relu:
    t4_load_pair
    v_max_f32_e64 v1, s20, 0.0
    t4_store

t4_clip:
    t4_load_pair
    v_max_f32_e64 v1, s20, 0.0
    v_min_f32_e64 v1, v1, 1.0
    t4_store

# G47 biasadd, one group per cell: g = i*cols + j, j = g % cols.
# Header [cols, pad, a_lo, a_hi, bias_lo, bias_hi], groups = rows*cols.
t4_biasadd:
    v_mov_b32 v6, v0
    s_load_dword s22, s[2:3], 0
    s_add_u32 s18, s2, 8
    s_addc_u32 s19, s3, 0
    s_load_dword s26, s[18:19], 0
    s_load_dword s27, s[18:19], 4
    s_load_dword s28, s[18:19], 8
    s_load_dword s29, s[18:19], 12
    s_waitcnt lgkmcnt(0)
    s_mov_b32 s30, s6
.Lbias_div:
    s_cmp_ge_u32 s30, s22
    s_cbranch_scc0 .Lbias_div_done
    s_sub_u32 s30, s30, s22
    s_branch .Lbias_div
.Lbias_div_done:
    s_lshl_b32 s16, s30, 2
    s_add_u32 s24, s28, s16
    s_addc_u32 s25, s29, 0
    s_lshl_b32 s16, s6, 2
    s_add_u32 s20, s26, s16
    s_addc_u32 s21, s27, 0
    s_add_u32 s18, s4, s16
    s_addc_u32 s19, s5, 0
    s_load_dword s14, s[20:21], 0
    s_load_dword s15, s[24:25], 0
    s_waitcnt lgkmcnt(0)
    v_add_f32_e64 v1, s14, s15
    v_mov_b32 v2, s18
    v_mov_b32 v3, s19
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm

# G48 matmul, one group per cell: g = i*N + j, i = g / N, j = g % N.
# Header [K, N, a_lo, a_hi, b_lo, b_hi], groups = rows*N.
t4_matmul:
    v_mov_b32 v6, v0
    s_load_dword s22, s[2:3], 0
    s_load_dword s23, s[2:3], 4
    s_add_u32 s18, s2, 8
    s_addc_u32 s19, s3, 0
    s_load_dword s26, s[18:19], 0
    s_load_dword s27, s[18:19], 4
    s_load_dword s28, s[18:19], 8
    s_load_dword s29, s[18:19], 12
    s_waitcnt lgkmcnt(0)
    s_mov_b32 s30, s6
    s_mov_b32 s31, 0
.Lmat_div:
    s_cmp_ge_u32 s30, s23
    s_cbranch_scc0 .Lmat_div_done
    s_sub_u32 s30, s30, s23
    s_add_u32 s31, s31, 1
    s_branch .Lmat_div
.Lmat_div_done:
    s_mul_i32 s16, s31, s22
    s_lshl_b32 s16, s16, 2
    s_add_u32 s20, s26, s16
    s_addc_u32 s21, s27, 0
    s_lshl_b32 s16, s30, 2
    s_add_u32 s24, s28, s16
    s_addc_u32 s25, s29, 0
    s_lshl_b32 s16, s6, 2
    s_add_u32 s18, s4, s16
    s_addc_u32 s19, s5, 0
    s_lshl_b32 s15, s23, 2
    v_mov_b32 v1, 0
    s_mov_b32 s31, s22
.Lmat_k:
    s_load_dword s13, s[20:21], 0
    s_load_dword s12, s[24:25], 0
    s_waitcnt lgkmcnt(0)
    v_mul_f32_e64 v2, s13, s12
    v_add_f32_e64 v1, v1, v2
    s_add_u32 s20, s20, 4
    s_addc_u32 s21, s21, 0
    s_add_u32 s24, s24, s15
    s_addc_u32 s25, s25, 0
    s_sub_u32 s31, s31, 1
    s_cmp_lg_u32 s31, 0
    s_cbranch_scc1 .Lmat_k
    v_mov_b32 v2, s18
    v_mov_b32 v3, s19
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm
