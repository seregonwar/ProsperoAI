# shotgun.s — PAI-M0 experiment F5 (shotgun value test)
#
# The store reads some register depending on kernel state; place the
# value in v0, v4 and v5 simultaneously to discover the source.
# c[i] == VAL means one of them flowed through; VAL+3 identifies v5+3.
# User data (RSRC2 0x08): s0 = VAL, s2:s3 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj shotgun.s -o shotgun.o
#   llvm-objcopy --dump-section .text=shotgun.bin shotgun.o

.text
.globl shotgun
shotgun:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_mov_b32 v4, s0
    v_mov_b32 v0, s0
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v5, v4
    flat_store_dword v[2:3], v4
    s_endpgm
