# dsprobe.s - PAI-M0 experiment G25
#
# Minimal LDS roundtrip: ds_write 0xDEAD0001 at LDS[0], ds_read it
# back, store via the G16 v0-operand pattern. Bisects the G23 hang:
# are the ds ops themselves broken?
# User data: s2:s3 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj dsprobe.s -o dsprobe.o
#   llvm-objcopy --dump-section .text=dsprobe.bin dsprobe.o

.text
.globl dsprobe

dsprobe:
    v_mov_b32 v1, 0x0
    v_mov_b32 v2, 0xDEAD0001
    ds_write_b32 v1, v2
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v4, v0
    v_lshlrev_b32 v4, 2, v4
    ds_read_b32 v6, v4
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v6
    flat_store_dword v[2:3], v0
    s_endpgm
