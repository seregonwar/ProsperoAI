# gemv_serial_u32.s - PAI-M1D serial-per-row GEMV correctness primitive
#
# NOT a performance kernel. Each workgroup is an independent
# dot_serial_u32 over one matrix row (TGID_X = row g).
#
#   groups_x = M, NUM_THREAD_X = 1
#   group g:
#       acc = 0
#       for k in 0 .. K-1:
#           acc += W[g, k] * x[k]     # uint32 wrap
#       y[g] = acc
#
# Two garlic buffers, G22 user-data ABI unchanged (s2:s3 = W base,
# s4:s5 = y, s6 = TGID_X, RSRC2 0x8C). x lives in a separate buffer;
# its 64-bit address is in the W header and is loaded with the proven
# immediate s_load 0/4 after a FREE-SGPR +8 shift (no extra user SGPR,
# no s_load SGPR-offset).
#
# W buffer layout (dword):
#   [0] = K
#   [1] = pad
#   [2] = x_lo
#   [3] = x_hi
#   [4 + g*K + k] = W[g, k]     # row-major, stride = K dwords
#
# x buffer: dword[k] = x[k]
# y buffer: dword[g] = y[g]
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj \
#       gemv_serial_u32.s -o gemv_serial_u32.o
#   llvm-objcopy --dump-section .text=gemv_serial_u32.bin gemv_serial_u32.o

.text
.globl gemv_serial_u32

gemv_serial_u32:
    s_load_dword s17, s[2:3], 0
    s_add_u32 s18, s2, 8
    s_addc_u32 s19, s3, 0
    s_load_dword s26, s[18:19], 0
    s_load_dword s27, s[18:19], 4
    s_waitcnt lgkmcnt(0)

    s_mul_i32 s16, s6, s17
    s_lshl_b32 s16, s16, 2
    s_add_u32 s20, s2, 16
    s_addc_u32 s21, s3, 0
    s_add_u32 s20, s20, s16
    s_addc_u32 s21, s21, 0

    s_mov_b32 s18, s26
    s_mov_b32 s19, s27
    s_mov_b32 s22, s0
    s_cmp_eq_u32 s17, 0
    s_cbranch_scc1 .Ldone

.Lloop:
    s_load_dword s24, s[20:21], 0
    s_load_dword s25, s[18:19], 0
    s_waitcnt lgkmcnt(0)
    s_mul_i32 s24, s24, s25
    s_add_u32 s22, s22, s24
    s_add_u32 s20, s20, 4
    s_addc_u32 s21, s21, 0
    s_add_u32 s18, s18, 4
    s_addc_u32 s19, s19, 0
    s_sub_u32 s17, s17, 1
    s_cmp_lg_u32 s17, 0
    s_cbranch_scc1 .Lloop

.Ldone:
    s_lshl_b32 s16, s6, 2
    s_add_u32 s18, s4, s16
    s_addc_u32 s19, s5, 0
    v_mov_b32 v2, s18
    v_mov_b32 v3, s19
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v1, vcc_lo, v0, s22
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm
