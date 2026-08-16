# gbatch2.s — PAI-M0 experiments G9-G12 (store data source probes)
#
# G9:  E32 replica — v0 = literal const, addr = C + tid*16, x4 store.
# G10: addr = C + tid*4 + 0x100      (mask probe)
# G11: addr = C + tid*4 + 0x1000000  (large offset probe)
# G12: F2 replica — v0 = tid, addr = C + tid*4, dword store (baseline)
# User data (RSRC2 0x08): s2:s3 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj gbatch2.s -o gbatch2.o
#   llvm-objcopy --dump-section .text=gbatch2.bin gbatch2.o

.text
.globl g9
.globl g10
.globl g11
.globl g12

g9:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 4, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, 0x12345678
    flat_store_dwordx4 v[2:3], v[4:7]
    s_endpgm

g10:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v2, vcc_lo, v2, 0x100
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_store_dword v[2:3], v4
    s_endpgm

g11:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v2, vcc_lo, v2, 0x1000000
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_store_dword v[2:3], v4
    s_endpgm

g12:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_mov_b32 v4, s0
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_f32 v1, v1, v4
    flat_store_dword v[2:3], v4
    s_endpgm
