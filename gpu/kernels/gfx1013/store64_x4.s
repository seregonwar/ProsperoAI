# store64_x4.s — PAI-M0 experiment E13
#
# 64 threads, one group; each thread stores 16 bytes of the constant
# 0xF00DFEED to dst + tid*16 using FLAT_STORE_DWORDX4 (the same store
# variant the OpenAGC golden kernel uses, op 0x78).
# User data (psbc ABI): s0-s1 = ring offsets, s2:s3 = dst.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj store64_x4.s -o store64_x4.o
#   llvm-objcopy --dump-section .text=store64_x4.bin store64_x4.o

.text
.globl store64_x4
store64_x4:
    v_mov_b32 v1, s2
    v_mov_b32 v2, s3
    v_mov_b32 v3, v0
    v_lshlrev_b32 v3, 4, v3
    v_add_co_u32 v4, vcc_lo, v1, v3
    v_add_co_ci_u32 v5, vcc_lo, v2, 0, vcc_lo
    v_mov_b32 v6, 0xF00DFEED
    v_mov_b32 v7, v6
    v_mov_b32 v8, v6
    v_mov_b32 v9, v6
    flat_store_dwordx4 v[4:5], v[6:9]
    s_endpgm
