# gbatch.s — PAI-M0 experiments G7/G8
#
# G7: x4 broadcast store replay — v0 = tid per-thread, 16-byte stride.
#     Expect c[4i..4i+3] = {i,i,i,i} (float-bit tid ramp).
# G8: per-thread INTEGER arithmetic: v6 = tid*4 + k (add_co with SGPR
#     src1, VOP3 SGPR reads proven), v0 = v6, x4 broadcast store.
#     Expect c[4i..4i+3] = tid*4 + k_int.
# User data (RSRC2 0x0C): s2:s3 = C, s4 = k (u32).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj gbatch.s -o gbatch.o
#   llvm-objcopy --dump-section .text=gbatch.bin gbatch.o

.text
.globl g7
.globl g8

# v0 = tid (hardware), v5 = tid*4, v[2:3] = C + tid*16, x4 store
g7:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 4, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_store_dwordx4 v[2:3], v[4:7]
    s_endpgm

# v6 = tid*4 + s4 (int, per-thread), v0 = v6, x4 broadcast store
g8:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 4, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_lshlrev_b32 v6, 2, v1
    v_add_co_u32 v6, vcc_lo, v6, s4
    v_mov_b32 v0, v6
    flat_store_dwordx4 v[2:3], v[4:7]
    s_endpgm
