# store_const64.s — PAI-M0 experiment E5/E7
#
# 64 threads, one group; each wave stores the constant 0xCAFEF00D to
# dst + tid*4 (idempotent across the two wave32s of the group).
# User data follows the psbc ABI: s0-s1 = ring offsets (unused),
# s2:s3 = dst (64-bit GPU VA).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj store_const64.s -o store_const64.o
#   llvm-objcopy --dump-section .text=store_const64.bin store_const64.o

.text
.globl store_const64
store_const64:
    v_mov_b32 v1, s2
    v_mov_b32 v2, s3
    v_mov_b32 v3, v0
    v_lshlrev_b32 v3, 2, v3
    v_add_co_u32 v4, vcc_lo, v1, v3
    v_add_co_ci_u32 v5, vcc_lo, v2, 0, vcc_lo
    v_mov_b32 v6, 0xCAFEF00D
    flat_store_dword v[4:5], v6
    s_endpgm
