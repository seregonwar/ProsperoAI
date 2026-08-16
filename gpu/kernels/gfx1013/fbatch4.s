# fbatch4.s - PAI-M0 experiment G33 (float vecadd form)
#
# The unlocked float pattern: cvt the tid, move the SGPR scalar k into
# a VGPR, per-thread float add with VGPR operands, dst != v0.
# Expect: c[i] = (float)(4i+3) + k.
# User data (RSRC2 0x0C): s2:s3 = C, s4 = k (float).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj fbatch4.s -o fbatch4.o
#   llvm-objcopy --dump-section .text=fbatch4.bin fbatch4.o

.text
.globl g33

g33:
    v_mov_b32 v6, v0
    v_cvt_f32_i32 v1, v0
    v_mov_b32 v7, s4
    v_add_f32_e64 v1, v1, v7
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm
