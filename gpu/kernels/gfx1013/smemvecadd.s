# smemvecadd.s - PAI-M0 experiment G23
#
# Scalar A[0]+B[0] via the G22-proven path. The harness packs one
# (A,B) pair at an already-shifted base; the kernel uses IMMEDIATE
# s_load offsets (0 and 4) — no SGPR offset, the form that hung the
# first G23. Destinations are s8/s9 (G22 wrote s8). Store is the G22
# formula with k = A+B, C at s4:s5, RSRC2 = 6 user SGPRs + G25 LDS bit.
#
# User data: s2:s3 = packed (A,B) base, s4:s5 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj smemvecadd.s -o smemvecadd.o
#   llvm-objcopy --dump-section .text=smemvecadd.bin smemvecadd.o

.text
.globl smemvecadd

smemvecadd:
    s_load_dword s8, s[2:3], 0
    s_load_dword s9, s[2:3], 4
    s_waitcnt lgkmcnt(0)
    s_add_u32 s8, s8, s9
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v1, vcc_lo, v0, s8
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm
