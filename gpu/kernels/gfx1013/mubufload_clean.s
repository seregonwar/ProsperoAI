# mubufload_clean.s - PAI-M0 experiment G20
#
# Clean MUBUF load test: buffer_load_dword into v6 from T#(s[0:3]),
# then store v6 to C[tid+64] with the proven G14-style store.
# The watch fill writes 0xA5A5A5A5 across C, so a working MUBUF read
# path stores 0xA5A5A5A5 to C[64..67].
# User data: s0:s3 = T#, s4:s5 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj mubufload_clean.s -o mubufload_clean.o
#   llvm-objcopy --dump-section .text=mubufload_clean.bin mubufload_clean.o

.text
.globl mubufload_clean

mubufload_clean:
    v_mov_b32 v1, v0
    v_lshlrev_b32 v1, 2, v1
    buffer_load_dword v6, v1, s[0:3], 0 offen
    s_waitcnt vmcnt(0)
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v2, vcc_lo, v2, 256
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v6
    flat_store_dword v[2:3], v0
    s_endpgm
