# dsprobe.s - PAI-M0 G25 LDS roundtrip (m0 init)
#
# Same G22 1-thread ABI (s2:s3 = C). Init m0=0 before DS.
# Store puts the LDS dword in v0 (the 9.40 flat_store data channel).
#
# OpenAGC gfx1013 note: LDS is allocated in 1 KiB blocks; the RSRC2
# field is still in 512-byte granules, so the minimum legal size is
# LDS_SIZE=2 (see OpenAGC HS path). Probe that size from the harness.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj dsprobe.s -o dsprobe.o
#   llvm-objcopy --dump-section .text=dsprobe.bin dsprobe.o

.text
.globl dsprobe

dsprobe:
    s_mov_b32 m0, 0
    v_mov_b32 v1, 0x0
    v_mov_b32 v2, 0xDEAD0001
    ds_write_b32 v1, v2
    s_waitcnt lgkmcnt(0)
    ds_read_b32 v6, v1
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v6
    flat_store_dword v[2:3], v0
    s_endpgm
