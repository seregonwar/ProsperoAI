# smemload.s - PAI-M0 experiment G19
#
# Scalar load from C[0] (CPU pre-filled with 0xA5A5A5A5), then every
# thread stores the loaded value to C[tid+64]. The SMEM/scalar read
# path is a different unit from the flat path - if this completes,
# only the vector read path is blocked.
# User data: s2:s3 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj smemload.s -o smemload.o
#   llvm-objcopy --dump-section .text=smemload.bin smemload.o

.text
.globl smemload

smemload:
    s_load_dword s8, s[2:3], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v2, vcc_lo, v2, 256
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, s8
    flat_store_dword v[2:3], v0
    s_endpgm
