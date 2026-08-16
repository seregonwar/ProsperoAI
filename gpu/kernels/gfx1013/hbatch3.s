# hbatch3.s — PAI-M0 experiment H10 (SMEM load breakthrough attempt)
#
# s_load_dwordx4 s[0:3], s[4:5], 0 loads A[0..3] (scalar), then the
# store writes (last SGPR value = s0 = A[0]) + addr_offset + 3.
# Expected: c[i] = A[0] + 4i + 3 for lanes 0-7.
# User data (RSRC2 0x0C): s2:s3 = C, s4:s5 = A (SMEM sbase pair).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj hbatch3.s -o hbatch3.o
#   llvm-objcopy --dump-section .text=hbatch3.bin hbatch3.o

.text
.globl h10
h10:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    s_load_dwordx4 s[0:3], s[4:5], 0
    s_waitcnt lgkmcnt(0)
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v6, vcc_lo, v5, s0
    flat_store_dword v[2:3], v4
    s_endpgm
