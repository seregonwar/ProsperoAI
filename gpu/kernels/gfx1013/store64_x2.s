# store64_x2.s — PAI-M0 experiment E12
#
# 64 threads, one group; each thread stores {0xBEADF00D, 0xBEADF00D}
# (two dwords) to dst + tid*8 using FLAT_STORE_DWORDX2.
# Hypothesis: PS5 gfx1013 implements only the X2/X4 flat store variants;
# the single-dword flat_store_dword (op 0x70) hangs the wave.
# User data (psbc ABI): s0-s1 = ring offsets, s2:s3 = dst.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj store64_x2.s -o store64_x2.o
#   llvm-objcopy --dump-section .text=store64_x2.bin store64_x2.o

.text
.globl store64_x2
store64_x2:
    v_mov_b32 v1, s2
    v_mov_b32 v2, s3
    v_mov_b32 v3, v0
    v_lshlrev_b32 v3, 3, v3
    v_add_co_u32 v4, vcc_lo, v1, v3
    v_add_co_ci_u32 v5, vcc_lo, v2, 0, vcc_lo
    v_mov_b32 v6, 0xBEADF00D
    v_mov_b32 v7, v6
    flat_store_dwordx2 v[4:5], v[6:7]
    s_endpgm
