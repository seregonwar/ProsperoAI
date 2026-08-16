# smemload_g15.s - PAI-M0 experiment G22
#
# SMEM scalar load of C[0] (the 0xA5 watch fill) -> s8, then the
# G15-proven store formula: c[i] = 4*i + s8 + 3. If the SMEM read
# returns the real fill, c[0] = 0xA5A5A5A8.
# User data: s2:s3 = C (load base), s4:s5 = C (store base).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj smemload_g15.s -o smemload_g15.o
#   llvm-objcopy --dump-section .text=smemload_g15.bin smemload_g15.o

.text
.globl smemload_g15

smemload_g15:
    s_load_dword s8, s[2:3], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v1, vcc_lo, v0, s8
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm
