# mubufload36.s - PAI-M0 experiment G36 (MUBUF with T# in non-zeroed SGPRs)
#
# The G20/G21 MUBUF tests put the T# in s[0:3] - but s0-s1 are the
# hardware-zeroed ring-offset slots! The T# base was 0, so every load
# hit VA 0 (zero-fill). Here the T# lives in s[4:7] (4-aligned) and C
# is at s2:s3. RSRC2 = 0x10 (4 pairs: s0-s7). readfirstlane target s8
# is outside the user window (free SGPR).
# The G15 store formula: c[i] = 4i + readfirstlane(v6) + 3.
# A is pre-filled 0xA5A5A5A5; a working read gives c[0] = 0xA5A5A5A8.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj \
#       mubufload36.s -o mubufload36.o
#   llvm-objcopy --dump-section .text=mubufload36.bin mubufload36.o

.text
.globl mubufload36

mubufload36:
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