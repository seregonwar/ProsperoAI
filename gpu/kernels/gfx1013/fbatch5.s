# fbatch5.s - PAI-M0 experiments G34/G35 (SGPR scalar into float add)
#
# G34: SGPR -> v0 (reliable dst-v0 read) -> v7 (VGPR-VGPR) -> e64 add.
# G35: direct e64 mixed add (v1 + s4) - does VOP3 read s4 with a
#      per-thread VGPR operand alive?
# Expect both: c[i] = (float)(4i+3) + k.
# User data (RSRC2 0x0C): s2:s3 = C, s4 = k (float).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj fbatch5.s -o fbatch5.o
#   llvm-objcopy --dump-section .text=fbatch5.bin fbatch5.o

.text
.globl g34
.globl g35

g34:
    v_mov_b32 v6, v0
    v_cvt_f32_i32 v1, v0
    v_mov_b32 v0, s4
    v_mov_b32 v7, v0
    v_add_f32_e64 v1, v1, v7
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm

g35:
    v_mov_b32 v6, v0
    v_cvt_f32_i32 v1, v0
    v_add_f32_e64 v1, v1, s4
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm