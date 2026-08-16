# mubufload_g15.s - PAI-M0 experiment G21
#
# MUBUF load c[tid] -> v6, readfirstlane into s8, then the G15-PROVEN
# store formula: c[i] = 4*i + s8 + 3. If the MUBUF read returns the
# real 0xA5A5A5A5 fill, c[0] = 0xA5A5A5A8; a broken read yields 3.
# User data: s0:s3 = T#, s4:s5 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj mubufload_g15.s -o mubufload_g15.o
#   llvm-objcopy --dump-section .text=mubufload_g15.bin mubufload_g15.o

.text
.globl mubufload_g15

mubufload_g15:
    v_mov_b32 v1, v0
    v_lshlrev_b32 v1, 2, v1
    buffer_load_dword v6, v1, s[0:3], 0 offen
    s_waitcnt vmcnt(0)
    v_readfirstlane_b32 s8, v6
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v1, vcc_lo, v0, s8
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm
