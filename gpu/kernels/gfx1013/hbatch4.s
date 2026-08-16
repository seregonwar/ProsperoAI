# hbatch4.s — PAI-M0 experiment H11 (SMEM load with the proven sbase pair)
#
# s_load_dwordx4 s[0:3], s[2:3], 0 loads A[0..3] using the SGPR pair
# that demonstrably carries user data (s2:s3). The store then writes
# (last SGPR value = s0 = A[0]) + addr_offset + 3, in place on A.
# Expected: A[i] = A[0]_original + 4i + 3 for lanes 0-7.
# User data (RSRC2 0x08): s2:s3 = A (load source AND store target).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj hbatch4.s -o hbatch4.o
#   llvm-objcopy --dump-section .text=hbatch4.bin hbatch4.o

.text
.globl h11
h11:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    s_load_dwordx4 s[0:3], s[2:3], 0
    s_waitcnt lgkmcnt(0)
    v_add_co_u32 v6, vcc_lo, v5, s0
    flat_store_dword v[2:3], v4
    s_endpgm
