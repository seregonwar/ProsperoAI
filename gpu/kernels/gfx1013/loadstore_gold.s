# loadstore_gold.s — PAI-M0 experiment E31
#
# Copy kernel with the golden register layout: every flat op uses
# address pair v[2:3] and data v[4:7] (vaddr pair 1).
# A[tid] -> C[tid], 32 threads.
# User data: s0-s1 ring offsets, s2:s3 = A, s4:s5 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj loadstore_gold.s -o loadstore_gold.o
#   llvm-objcopy --dump-section .text=loadstore_gold.bin loadstore_gold.o

.text
.globl loadstore_gold
loadstore_gold:
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_mov_b32 v4, s4
    v_mov_b32 v5, s5
    v_mov_b32 v1, v0
    v_lshlrev_b32 v1, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v1
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_load_dword v6, v[2:3]
    v_add_co_u32 v4, vcc_lo, v4, v1
    v_add_co_ci_u32 v5, vcc_lo, v5, 0, vcc_lo
    s_waitcnt vmcnt(0) lgkmcnt(0)
    flat_store_dword v[4:5], v6
    s_endpgm
