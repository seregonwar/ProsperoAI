# vecadd.s — PAI-M0 first gfx1013 compute kernel
#
# c[i] = a[i] + b[i], f32, single group of 32 threads (W32 mode).
# Launch: NUM_THREAD_X = 32, 1 group. n is clamped to 32 by the harness.
#
# User data ABI (COMPUTE_USER_DATA_0..6):
#   s0:s1 = A (64-bit GPU VA)
#   s2:s3 = B
#   s4:s5 = C
#   s6    = n (u32, unused by the kernel body)
#
# Assemble (WSL or any llvm-mc >= 14):
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj vecadd.s -o vecadd.o
#   llvm-objcopy --dump-section .text=vecadd.bin vecadd.o

.text
.globl vecadd
vecadd:
    v_mov_b32 v1, s0
    v_mov_b32 v2, s1
    v_mov_b32 v3, s2
    v_mov_b32 v4, s3
    v_mov_b32 v5, s4
    v_mov_b32 v6, s5
    v_mov_b32 v7, v0
    v_lshlrev_b32 v8, 2, v7
    v_add_co_u32 v9, vcc_lo, v1, v8
    v_add_co_ci_u32 v10, vcc_lo, v2, 0, vcc_lo
    flat_load_dword v11, v[9:10]
    v_add_co_u32 v12, vcc_lo, v3, v8
    v_add_co_ci_u32 v13, vcc_lo, v4, 0, vcc_lo
    flat_load_dword v14, v[12:13]
    s_waitcnt vmcnt(0) lgkmcnt(0)
    v_add_f32 v15, v11, v14
    v_add_co_u32 v16, vcc_lo, v5, v8
    v_add_co_ci_u32 v17, vcc_lo, v6, 0, vcc_lo
    flat_store_dword v[16:17], v15
    s_endpgm
