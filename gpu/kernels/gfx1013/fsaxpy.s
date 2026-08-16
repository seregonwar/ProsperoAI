# fsaxpy.s - PAI-M0 experiment G41 (VALU float SAXPY, per-group)
#
# Float32 SAXPY clone of saxpy.s (M1-S validated): C[g] = a*A[g] + B[g]
# with groups_x = N, NUM_THREAD_X = 1, TGID_X = g. The float scalar
# `a` is a VALU immediate (0.5f = 0x3F000000) and the ops are the
# unlocked e64 forms (direct-SGPR operand + literal / VGPR+SGPR).
#
# Packed input: dword[2*g]=A[g], dword[2*g+1]=B[g].
# User data: s2:s3 = packed AB, s4:s5 = C, s6 = TGID_X. RSRC2 = 0x8C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj fsaxpy.s -o fsaxpy.o
#   llvm-objcopy --dump-section .text=fsaxpy.bin fsaxpy.o

.text
.globl fsaxpy

fsaxpy:
    s_lshl_b32 s16, s6, 3
    s_add_u32 s18, s2, s16
    s_addc_u32 s19, s3, 0
    s_load_dword s20, s[18:19], 0
    s_load_dword s21, s[18:19], 4
    s_waitcnt lgkmcnt(0)
    v_mul_f32_e64 v1, s20, 0.5
    v_add_f32_e64 v1, v1, s21
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