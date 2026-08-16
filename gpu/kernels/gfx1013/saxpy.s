# saxpy.s - PAI-M0 integer SAXPY: C[i] = a * A[i] + B[i]
#
# Same queue/dispatch/resources as add1d.s (G22 PM4, TGID_X = i,
# NUM_THREAD_X = 1, groups_x = N). Scalar `a` is an immediate (a = 3)
# via s_mulk_i32 so the proven USER_SGPR=6 + TGID_X_EN ABI is unchanged:
#   s2:s3 = packed AB base, s4:s5 = C, s6 = TGID_X
# Packed input: dword[2*i]=A[i], dword[2*i+1]=B[i].
# Base is shifted in FREE SGPRs; s_load uses IMMEDIATE offset 0 / 4.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj saxpy.s -o saxpy.o
#   llvm-objcopy --dump-section .text=saxpy.bin saxpy.o

.text
.globl saxpy

saxpy:
    s_lshl_b32 s16, s6, 3
    s_add_u32 s18, s2, s16
    s_addc_u32 s19, s3, 0
    s_load_dword s20, s[18:19], 0
    s_load_dword s21, s[18:19], 4
    s_waitcnt lgkmcnt(0)
    s_mulk_i32 s20, 3
    s_add_u32 s20, s20, s21
    s_lshl_b32 s16, s6, 2
    s_add_u32 s18, s4, s16
    s_addc_u32 s19, s5, 0
    v_mov_b32 v2, s18
    v_mov_b32 v3, s19
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v1, vcc_lo, v0, s20
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm
