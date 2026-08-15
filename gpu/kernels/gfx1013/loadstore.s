# loadstore.s — PAI-M0 experiment E2
#
# 32 threads; c[tid] = a[tid] through flat load + flat store.
# Validates the flat_load encoding on gfx1013.
# User data: s0:s1 = A, s2:s3 = C (64-bit GPU VAs)
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj loadstore.s -o loadstore.o
#   llvm-objcopy --dump-section .text=loadstore.bin loadstore.o

.text
.globl loadstore
loadstore:
    v_mov_b32 v1, s0
    v_mov_b32 v2, s1
    v_mov_b32 v3, s2
    v_mov_b32 v4, s3
    v_mov_b32 v5, v0
    v_lshlrev_b32 v5, 2, v5
    v_add_co_u32 v6, vcc_lo, v1, v5
    v_add_co_ci_u32 v7, vcc_lo, v2, 0, vcc_lo
    flat_load_dword v8, v[6:7]
    v_add_co_u32 v9, vcc_lo, v3, v5
    v_add_co_ci_u32 v10, vcc_lo, v4, 0, vcc_lo
    s_waitcnt vmcnt(0) lgkmcnt(0)
    flat_store_dword v[9:10], v8
    s_endpgm
