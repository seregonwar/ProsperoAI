# arith4.s — PAI-M0 experiment E47 (THE milestone kernel)
#
# c[i] = (float)i + k for i < 32. RSRC2 0x08 config.
# Rule learned on 9.40: VOP3 with mixed VGPR+SGPR operands reads the
# VGPR as 0; VOP1 SGPR reads work. So k moves s0 -> v4 first, then the
# add uses VGPR+VGPR.
# User data: s0 = k, s1 = 0, s2:s3 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj arith4.s -o arith4.o
#   llvm-objcopy --dump-section .text=arith4.bin arith4.o

.text
.globl arith4
.globl arith4b

arith4:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_mov_b32 v4, s0
    v_add_f32 v0, v1, v4
    v_lshlrev_b32 v1, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v1
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_store_dword v[2:3], v4
    s_endpgm

arith4b:
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_mov_b32 v4, s0
    v_add_f32 v0, v0, v4
    v_lshlrev_b32 v1, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v1
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_store_dword v[2:3], v4
    s_endpgm
