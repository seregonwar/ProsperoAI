# cossin.s - PAI-M0 experiments G65/G66 (wave-parallel cos/sin ramp)
#
# RoPE position-table primitives (Phase 2): validate the VALU cos/sin
# instructions on 9.40 with the proven wave-parallel value path (G55/
# G56 ramp): NUM_THREAD_X=32, lanes 0..7 store, theta derived
# arithmetically from tid + uniform scale.
#
#   G65 c[i] = cos(scale * i)     (i = lane id, value-path (4i+3))
#   G66 c[i] = sin(scale * i)
#
# The value-path quirk (G35/G55) means the lane index read through
# v_cvt_f32_i32(v0) is (4i+3), not i; the oracle uses the same
# convention: want[i] = cosf(scale*(4i+3)).
#
# VALU form: v_cvt_f32_i32 (int->float), v_mul_f32_e64 with direct
# SGPR operand (G35/G39 rules), v_cos_f32/v_sin_f32 VOP2 with dst v1
# (dst != v0), final v0 copy for the store.
#
# User data ABI (RSRC2 0x0C, same as G33/G34/G35):
#   s2:s3 = C (64-bit GPU VA)
#   s4    = scale (float bits)
# Free SGPRs s16+, VGPRs: v1 acc, v6 = saved tid.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj cossin.s -o cossin.o
#   llvm-objcopy --dump-section .text=cossin.bin cossin.o

.text
.globl cossin_cos
.globl cossin_sin

cossin_cos:
    v_mov_b32 v6, v0
    v_cvt_f32_i32 v1, v0
    v_mul_f32_e64 v1, v1, s4
    v_cos_f32 v1, v1
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm

cossin_sin:
    v_mov_b32 v6, v0
    v_cvt_f32_i32 v1, v0
    v_mul_f32_e64 v1, v1, s4
    v_sin_f32 v1, v1
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm
