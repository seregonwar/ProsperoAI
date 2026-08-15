# store_const.s — PAI-M0 experiment E1
#
# 32 threads; each stores the constant 0xABCD1234 to dst + tid*4.
# Minimal validated-encoding kernel: no loads, no branches.
# User data: s0:s1 = dst (64-bit GPU VA)
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj store_const.s -o store_const.o
#   llvm-objcopy --dump-section .text=store_const.bin store_const.o

.text
.globl store_const
store_const:
    v_mov_b32 v1, s0
    v_mov_b32 v2, s1
    v_mov_b32 v3, v0
    v_lshlrev_b32 v3, 2, v3
    v_add_co_u32 v4, vcc_lo, v1, v3
    v_add_co_ci_u32 v5, vcc_lo, v2, 0, vcc_lo
    v_mov_b32 v6, 0xABCD1234
    flat_store_dword v[4:5], v6
    s_endpgm
