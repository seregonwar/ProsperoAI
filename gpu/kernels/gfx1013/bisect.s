# bisect.s — PAI-M0 instruction bisection (E15-E19)
#
# Five kernels, each adding one instruction family. The first that stops
# executing on the PS5 pinpoints the toxic instruction.
#
# E15 bare:       v_mov_b32 + s_endpgm
# E16 bare_lshl:  + v_lshlrev_b32
# E17 bare_addco: + v_add_co_u32   (VOP3)
# E18 bare_addci: + v_add_co_ci_u32 (VOP3/E64)
# E19 bare_store: + flat_store_dwordx4 (memory op, golden-style)
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj bisect.s -o bisect.o
#   llvm-objcopy --dump-section .text=bisect.bin bisect.o

.text
.globl bare
.globl bare_lshl
.globl bare_addco
.globl bare_addci
.globl bare_store

bare:
    v_mov_b32 v1, s2
    s_endpgm

bare_lshl:
    v_mov_b32 v1, s2
    v_lshlrev_b32 v3, 2, v0
    s_endpgm

bare_addco:
    v_mov_b32 v1, s2
    v_lshlrev_b32 v3, 2, v0
    v_add_co_u32 v4, vcc_lo, v1, v3
    s_endpgm

bare_addci:
    v_mov_b32 v1, s2
    v_lshlrev_b32 v3, 2, v0
    v_add_co_u32 v4, vcc_lo, v1, v3
    v_add_co_ci_u32 v5, vcc_lo, s3, 0, vcc_lo
    s_endpgm

bare_store:
    v_mov_b32 v1, s2
    v_lshlrev_b32 v3, 2, v0
    v_add_co_u32 v4, vcc_lo, v1, v3
    v_add_co_ci_u32 v5, vcc_lo, s3, 0, vcc_lo
    v_mov_b32 v6, 0xDEADBEEF
    v_mov_b32 v7, v6
    v_mov_b32 v8, v6
    v_mov_b32 v9, v6
    flat_store_dwordx4 v[4:5], v[6:9]
    s_endpgm
