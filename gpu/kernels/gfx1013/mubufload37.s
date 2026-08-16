# mubufload37.s - PAI-M0 experiments G37/G38 (MUBUF format matrix)
#
# SRSRC in s[4:7], C at s2:s3, RSRC2 = 0x10. The G36 still zero-filled
# with word3 = 0x31014FAC (DATA_FORMAT = 1 = 8-bit!). G37/G38 fix the
# DATA_FORMAT to 32-bit (bits [21:16] = 4) and use the OpenAGC cache
# word2 = 0x20002000 instead of 4096.
# G37: word3 = 0x31044FAC (fmt 32, size 0x4FAC records)
# G38: word3 = 0x31040080 (fmt 32, size 128 records)
# c[0] = 0xA5A5A5A8 on a working read (0xA5A5A5A5 fill).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj \
#       mubufload37.s -o mubufload37.o
#   llvm-objcopy --dump-section .text=mubufload37.bin mubufload37.o

.text
.globl mubufload37
.globl mubufload38

mubufload37:
    v_mov_b32 v1, v0
    v_lshlrev_b32 v1, 2, v1
    buffer_load_dword v6, v1, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    v_readfirstlane_b32 s8, v6
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v1, vcc_lo, v0, s8
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm

mubufload38:
    v_mov_b32 v1, v0
    v_lshlrev_b32 v1, 2, v1
    buffer_load_dword v6, v1, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    v_readfirstlane_b32 s8, v6
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v1, vcc_lo, v0, s8
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm