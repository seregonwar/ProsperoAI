# ramp.s - PAI-M0 experiment G55 (wave-parallel float linear ramp)
#
# Gate: wave-parallel dispatch beyond the serial NUM_THREAD_X=1 model
# (whitepaper §43). NUM_THREAD_X = 32, one group; each lane derives its
# value arithmetically from tid (v0) + uniform scalars, the only reads
# that work on 9.40 (flat_load/MUBUF hang or zero-fill).
#
#   c[i] = base + k * i     (i = lane id, f32)
#
# VALU float form (G35/G39/G40 rules): v_cvt_f32_i32 (int->float),
# v_mul_f32_e64 with direct SGPR operand, v_add_f32_e64 VGPR+SGPR
# accumulator, dst != v0, final v0 copy for the store.
#
# User data ABI (RSRC2 0x0C, same as G33/G34/G35):
#   s2:s3 = C (64-bit GPU VA)
#   s4    = k (float bits)
#   s5    = base (float bits)
# Free SGPRs s16+, VGPRs: v1 acc, v6 = saved tid.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj ramp.s -o ramp.o
#   llvm-objcopy --dump-section .text=ramp.bin ramp.o

.text
.globl ramp

ramp:
    v_mov_b32 v6, v0
    v_cvt_f32_i32 v1, v0
    v_mul_f32_e64 v1, v1, s4
    v_add_f32_e64 v1, v1, s5
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm
