# smemload16.s - PAI-M0 experiment G24
#
# Only s_load_dwordx16 (16 dwords) + readfirstlane + the G15 store
# formula. Bisects the G23 hang: is the x16 scalar load itself broken?
# User data: s2:s3 = C, s4:s5 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj smemload16.s -o smemload16.o
#   llvm-objcopy --dump-section .text=smemload16.bin smemload16.o

.text
.globl smemload16

smemload16:
    s_load_dwordx16 s[16:31], s[2:3], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v1, vcc_lo, v0, s16
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm
