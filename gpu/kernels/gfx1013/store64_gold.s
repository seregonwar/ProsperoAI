# store64_gold.s — PAI-M0 experiment E30
#
# Store kernel with the golden kernel's exact register layout:
# address in v[2:3], data in v[4:7] (flat vaddr pair 1 — the only
# combination the 9.40 silicon accepts; bit15 irrelevant per G2).
# User data: s0-s1 ring offsets, s2:s3 = dst.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj store64_gold.s -o store64_gold.o
#   llvm-objcopy --dump-section .text=store64_gold.bin store64_gold.o

.text
.globl store64_gold
store64_gold:
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_mov_b32 v1, v0
    v_lshlrev_b32 v1, 4, v1
    v_add_co_u32 v2, vcc_lo, v2, v1
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v4, 0xB0DD00D1
    v_mov_b32 v5, v4
    v_mov_b32 v6, v4
    v_mov_b32 v7, v4
    flat_store_dwordx4 v[2:3], v[4:7]
    s_endpgm
