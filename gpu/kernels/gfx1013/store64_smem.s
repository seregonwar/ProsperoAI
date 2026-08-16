# store64_smem.s — PAI-M0 experiment E22
#
# E19 + the golden kernel's SMEM load as the first instruction:
#   s_load_dwordx4 s[0:3], s[2:3], 0x27C
# Hypothesis: flat ops on PS5 need the wave's SMEM to run first
# (or the load primes the flat addressing state).
# User data: s0-s1 ring offsets, s2:s3 = dst.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj store64_smem.s -o store64_smem.o
#   llvm-objcopy --dump-section .text=store64_smem.bin store64_smem.o

.text
.globl store64_smem
store64_smem:
    s_load_dwordx4 s[0:3], s[2:3], 0x27C
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v1, s2
    v_mov_b32 v2, s3
    v_mov_b32 v3, v0
    v_lshlrev_b32 v3, 4, v3
    v_add_co_u32 v4, vcc_lo, v1, v3
    v_add_co_ci_u32 v5, vcc_lo, v2, 0, vcc_lo
    v_mov_b32 v6, 0xC0FFEEEE
    v_mov_b32 v7, v6
    v_mov_b32 v8, v6
    v_mov_b32 v9, v6
    flat_store_dwordx4 v[4:5], v[6:9]
    s_endpgm
