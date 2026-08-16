# gbatch3.s — PAI-M0 experiments G13/G14
#
# G13: two literals (early A, late B) — which one does the store write?
# G14: literal + per-thread addresses (formal milestone fill)
# User data (RSRC2 0x08): s2:s3 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj gbatch3.s -o gbatch3.o
#   llvm-objcopy --dump-section .text=gbatch3.bin gbatch3.o

.text
.globl g13
.globl g14

g13:
    v_mov_b32 v1, 0xAAAA1111
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v4, 0xBBBB2222
    flat_store_dword v[2:3], v4
    s_endpgm

g14:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, 0xDEADBEEF
    flat_store_dword v[2:3], v4
    s_endpgm
