# selfref_nowait.s - PAI-M0 experiment G18
#
# G16 without any s_waitcnt: store 0x12345678 to C[tid], load it
# right back, store the loaded value to C[tid + 32]. If the label
# fires here, the WAITCNT was the hang, not the load itself.
# User data: s2:s3 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj selfref_nowait.s -o selfref_nowait.o
#   llvm-objcopy --dump-section .text=selfref_nowait.bin selfref_nowait.o

.text
.globl selfref_nowait

selfref_nowait:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, 0x12345678
    flat_store_dword v[2:3], v0
    flat_load_dword v6, v[2:3]
    v_add_co_u32 v7, vcc_lo, v2, 128
    v_add_co_ci_u32 v8, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v6
    flat_store_dword v[7:8], v0
    s_endpgm
