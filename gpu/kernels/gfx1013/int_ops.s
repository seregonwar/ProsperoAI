# int_ops.s - PAI-M0 experiments G49-G54 (T4 serial integer kernels)
#
# Integer mirror of t4_ops.s (G42-G48), same validated ABI:
#   s2:s3 = packed inputs / header, s4:s5 = C, s6 = TGID_X,
#   RSRC2 = 0x8C, groups_x = N (elementwise) or rows*cols (matmul),
#   1 thread per group. All arithmetic is u32 wrap (SALU integer
#   path proven in G22 / saxpy / gemv_serial_u32).
#
# Elementwise (G49-G53), packed (a,b) pairs at dword[2g], dword[2g+1]:
#   add2d: c[g] = a[g] + b[g]
#   sub1d: c[g] = a[g] - b[g]
#   mul1d: c[g] = a[g] * b[g]
#   relu:  c[g] = max(a[g], 0)
#   clip:  c[g] = clamp(a[g], 0, 1)
# Matmul (G54): c[g] = sum_k a[i*K+k] * b[k*N+j] (u32 wrap), one
#   group per cell: g = i*N + j, i = g / N, j = g % N.
#   Header [K, N, a_lo, a_hi, b_lo, b_hi], groups = rows*N.
#   b is row-major [k][N], so the k-step advances b by N*4 bytes.
#   HW rule: store only OUTSIDE loops (single final store per group).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj int_ops.s -o int_ops.o
#   llvm-objcopy --dump-section .text=int_ops.bin int_ops.o

.text
.globl int_add2d
.globl int_sub1d
.globl int_mul1d
.globl int_relu
.globl int_clip
.globl int_matmul

.macro int_store
    s_lshl_b32 s16, s6, 2
    s_add_u32 s18, s4, s16
    s_addc_u32 s19, s5, 0
    v_mov_b32 v2, s18
    v_mov_b32 v3, s19
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, s14
    flat_store_dword v[2:3], v0
    s_endpgm
.endm

.macro int_load_pair
    s_lshl_b32 s16, s6, 3
    s_add_u32 s18, s2, s16
    s_addc_u32 s19, s3, 0
    s_load_dword s20, s[18:19], 0
    s_load_dword s21, s[18:19], 4
    s_waitcnt lgkmcnt(0)
.endm

int_add2d:
    int_load_pair
    s_add_u32 s14, s20, s21
    int_store

int_sub1d:
    int_load_pair
    s_sub_u32 s14, s20, s21
    int_store

int_mul1d:
    int_load_pair
    s_mul_i32 s14, s20, s21
    int_store

int_relu:
    int_load_pair
    s_max_u32 s14, s20, 0
    int_store

int_clip:
    int_load_pair
    s_max_u32 s14, s20, 0
    s_min_u32 s14, s14, 1
    int_store

# G54 matmul, one group per cell: g = i*N + j, i = g / N, j = g % N.
# Header [K, N, a_lo, a_hi, b_lo, b_hi], groups = rows*N.
int_matmul:
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
.Lint_mat_div:
    s_cmp_ge_u32 s30, s23
    s_cbranch_scc0 .Lint_mat_div_done
    s_sub_u32 s30, s30, s23
    s_add_u32 s31, s31, 1
    s_branch .Lint_mat_div
.Lint_mat_div_done:
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
    s_mov_b32 s14, 0
    s_mov_b32 s31, s22
.Lint_mat_k:
    s_load_dword s13, s[20:21], 0
    s_load_dword s12, s[24:25], 0
    s_waitcnt lgkmcnt(0)
    s_mul_i32 s13, s13, s12
    s_add_u32 s14, s14, s13
    s_add_u32 s20, s20, 4
    s_addc_u32 s21, s21, 0
    s_add_u32 s24, s24, s15
    s_addc_u32 s25, s25, 0
    s_sub_u32 s31, s31, 1
    s_cmp_lg_u32 s31, 0
    s_cbranch_scc1 .Lint_mat_k
    v_mov_b32 v1, s14
    v_mov_b32 v2, s18
    v_mov_b32 v3, s19
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm
