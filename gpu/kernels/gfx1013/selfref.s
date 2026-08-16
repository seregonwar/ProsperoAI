# selfref.s - PAI-M0 experiment G16
#
# Store 0x12345678 to C[tid]; load it right back; store the loaded
# value to C[tid + 32]. Distinguishes "loads are broken at the
# instruction level" from "the VM/buffer the loads read is wrong":
# a load of a just-written address must return the written value.
# User data (RSRC2 0x08): s2:s3 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj selfref.s -o selfref.o
#   llvm-objcopy --dump-section .text=selfref.bin selfref.o

.text
.globl selfref

selfref:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, 0x12345678
    flat_store_dword v[2:3], v0
    s_waitcnt vmcnt(0)
    flat_load_dword v6, v[2:3]
    s_waitcnt vmcnt(0) lgkmcnt(0)
    v_add_co_u32 v7, vcc_lo, v2, 128
    v_add_co_ci_u32 v8, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v6
    flat_store_dword v[7:8], v0
    s_endpgm
