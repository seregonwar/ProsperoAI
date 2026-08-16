# mubufload.s — PAI-M0 experiment E38
#
# Per-thread loads via the MUBUF unit (buffer_load_dword) instead of
# the broken FLAT loads. T# descriptor for A in s[0:3], C base in s4:s5.
# Copy: c[i] = a[i] via buffer_load into v0 (store broadcasts v0).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj mubufload.s -o mubufload.o
#   llvm-objcopy --dump-section .text=mubufload.bin mubufload.o

.text
.globl mubufload_v0
mubufload_v0:
    v_mov_b32 v1, v0
    v_lshlrev_b32 v1, 2, v1
    buffer_load_dword v0, v1, s[0:3], 0 offen
    s_waitcnt vmcnt(0)
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_add_co_u32 v2, vcc_lo, v2, v1
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_store_dword v[2:3], v4
    s_endpgm
