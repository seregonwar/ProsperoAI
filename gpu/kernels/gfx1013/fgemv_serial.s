# fgemv_serial.s - PAI-M0 experiment G40 (VALU float GEMV, serial-per-row)
#
# Float32 GEMV clone of gemv_serial_u32 (M1D validated) with the
# unlocked VALU float form. groups_x = M, NUM_THREAD_X = 1, group g
# computes y[g] = sum_k W[g,k] * x[k] via the serial SMEM loop and
# the e64 direct-SGPR float mul + VGPR+VGPR accumulator add.
#
# W buffer layout (dword):
#   [0] = K, [1] = pad, [2] = x_lo, [3] = x_hi,
#   [4 + g*K + k] = W[g, k]
# x buffer: dword[k] = x[k]. y buffer: dword[g] = y[g].
# User data: s2:s3 = W, s4:s5 = y, s6 = TGID_X. RSRC2 = 0x8C.
# Free SGPRs s16-s27; VGPRs: v1 = acc, v3 = mul tmp.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj \
#       fgemv_serial.s -o fgemv_serial.o
#   llvm-objcopy --dump-section .text=fgemv_serial.bin fgemv_serial.o

.text
.globl fgemv_serial

fgemv_serial:
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
    v_mov_b32 v1, 0
    s_cmp_eq_u32 s17, 0
    s_cbranch_scc1 .Ldone

.Lloop:
    s_load_dword s24, s[20:21], 0
    s_load_dword s25, s[18:19], 0
    s_waitcnt lgkmcnt(0)
    v_mul_f32_e64 v3, s24, s25
    v_add_f32_e64 v1, v1, v3
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
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm