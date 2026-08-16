# acqload.s - PAI-M0 experiment G17
#
# Load one dword from the acqrb VA (the kernel's own GPU ring, passed
# in s2:s3) and store the loaded value to C[tid]. Distinguishes
# "loads hang everywhere" from "loads hang only on our dmem pages":
# the label fires iff the load completes.
# User data: s2:s3 = acqrb VA, s4:s5 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj acqload.s -o acqload.o
#   llvm-objcopy --dump-section .text=acqload.bin acqload.o

.text
.globl acqload

acqload:
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    flat_load_dword v6, v[2:3]
    s_waitcnt vmcnt(0) lgkmcnt(0)
    v_mov_b32 v1, v0
    v_mov_b32 v4, s4
    v_mov_b32 v5, s5
    v_lshlrev_b32 v7, 2, v1
    v_add_co_u32 v4, vcc_lo, v4, v7
    v_add_co_ci_u32 v5, vcc_lo, v5, 0, vcc_lo
    v_mov_b32 v0, v6
    flat_store_dword v[4:5], v0
    s_endpgm
