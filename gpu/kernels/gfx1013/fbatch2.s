# fbatch2.s - PAI-M0 experiments G28-G30 (float ALU v2)
#
# The old F-batch added floats on raw tid bits (denormals, flushed to
# 0 by the RSRC1 denorm mode). v0 = 4*tid+3 per the store empirics.
# New approach: convert the integer v0 to float first, then add.
# G28: cvt only (c[i] = (float)(4i+3))
# G29: cvt + v_add_f32_e64 (VOP3 SGPR read of s4 = k)
# G30: cvt + v_add_f32 e32 (VOP2 control)
# User data (RSRC2 0x0C): s2:s3 = C, s4 = k (float).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj fbatch2.s -o fbatch2.o
#   llvm-objcopy --dump-section .text=fbatch2.bin fbatch2.o

.text
.globl g28
.globl g29
.globl g30

g28:
    v_cvt_f32_i32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm

g29:
    v_cvt_f32_i32 v1, v0
    v_add_f32_e64 v1, v1, s4
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm

g30:
    v_cvt_f32_i32 v1, v0
    v_add_f32 v1, v1, s4
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm
