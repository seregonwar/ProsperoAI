# t4_ops.s - PAI-M0 experiments G42-G48 (T4 serial float kernels)
#
# All kernels use the validated e64 form (G35/G39/G40/G41):
# direct-SGPR operands, dst != v0, v0 = value for the store.
# ABI: s2:s3 = packed inputs, s4:s5 = C, s6 = TGID_X, RSRC2 = 0x8C,
# groups_x = N (elementwise) or rows (biasadd/matmul), 1 thread.
#
# Elementwise (G42-G46), packed (a,b) pairs at dword[2g], dword[2g+1]:
#   add1d: c[g] = a[g] + b[g]
#   sub1d: c[g] = a[g] - b[g]
#   mul1d: c[g] = a[g] * b[g]
#   relu:  c[g] = max(a[g], 0)
#   clip:  c[g] = clamp(a[g], 0, 1)
# Biasadd (G47): c[g*cols+j] = a[g*cols+j] + bias[j], header [cols, pad,
#   a_lo, a_hi, bias_lo, bias_hi], rows = groups.
# Matmul (G48): c[i*N+j] = sum_k a[i*K+k] * b[k*N+j], header [K, N,
#   a_lo, a_hi, b_lo, b_hi], rows = groups.
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

# elementwise common tail: v1 holds the value; v6 = g*4 for the vaddr.
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

# load the packed (a[g], b[g]) into s20/s21
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

t4_biasadd:
    s_load_dword s22, s[2:3], 0
    s_add_u32 s18, s2, 8
    s_addc_u32 s19, s3, 0
    s_load_dword s26, s[18:19], 0
    s_load_dword s27, s[18:19], 4
    s_load_dword s28, s[18:19], 8
    s_load_dword s29, s[18:19], 12
    s_waitcnt lgkmcnt(0)
    s_mul_i32 s16, s6, s22
    s_lshl_b32 s16, s16, 2
    s_add_u32 s20, s26, s16
    s_addc_u32 s21, s27, 0
    s_add_u32 s18, s4, s16
    s_addc_u32 s19, s5, 0
    s_mov_b32 s24, s28
    s_mov_b32 s25, s29
    s_mov_b32 s23, s22
.Lbias_j:
    s_load_dword s14, s[20:21], 0
    s_load_dword s15, s[24:25], 0
    s_waitcnt lgkmcnt(0)
    v_add_f32_e64 v1, s14, s15
    v_mov_b32 v2, s18
    v_mov_b32 v3, s19
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_add_u32 s20, s20, 4
    s_addc_u32 s21, s21, 0
    s_add_u32 s18, s18, 4
    s_addc_u32 s19, s19, 0
    s_add_u32 s24, s24, 4
    s_addc_u32 s25, s25, 0
    s_sub_u32 s23, s23, 1
    s_cmp_lg_u32 s23, 0
    s_cbranch_scc1 .Lbias_j
    s_endpgm

t4_matmul:
    s_load_dword s22, s[2:3], 0
    s_load_dword s23, s[2:3], 4
    s_lshl_b32 s15, s23, 2
    s_add_u32 s18, s2, 8
    s_addc_u32 s19, s3, 0
    s_load_dword s26, s[18:19], 0
    s_load_dword s27, s[18:19], 4
    s_load_dword s28, s[18:19], 8
    s_load_dword s29, s[18:19], 12
    s_waitcnt lgkmcnt(0)
    s_mul_i32 s16, s6, s22
    s_lshl_b32 s16, s16, 2
    s_add_u32 s20, s26, s16
    s_addc_u32 s21, s27, 0
    s_mov_b32 s24, s28
    s_mov_b32 s25, s29
    s_mul_i32 s16, s6, s23
    s_lshl_b32 s16, s16, 2
    s_add_u32 s18, s4, s16
    s_addc_u32 s19, s5, 0
    s_mov_b32 s30, s23
.Lmat_j:
    v_mov_b32 v1, 0
    s_mov_b32 s26, s20
    s_mov_b32 s27, s21
    s_mov_b32 s28, s24
    s_mov_b32 s29, s25
    s_mov_b32 s31, s22
.Lmat_k:
    s_load_dword s13, s[26:27], 0
    s_load_dword s12, s[28:29], 0
    s_waitcnt lgkmcnt(0)
    v_mul_f32_e64 v2, s13, s12
    v_add_f32_e64 v1, v1, v2
    s_add_u32 s26, s26, 4
    s_addc_u32 s27, s27, 0
    s_add_u32 s28, s28, s15
    s_addc_u32 s29, s29, 0
    s_sub_u32 s31, s31, 1
    s_cmp_lg_u32 s31, 0
    s_cbranch_scc1 .Lmat_k
    v_mov_b32 v2, s18
    v_mov_b32 v3, s19
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_add_u32 s18, s18, 4
    s_addc_u32 s19, s19, 0
    s_add_u32 s24, s24, 4
    s_addc_u32 s25, s25, 0
    s_sub_u32 s30, s30, 1
    s_cmp_lg_u32 s30, 0
    s_cbranch_scc1 .Lmat_j
    s_endpgm