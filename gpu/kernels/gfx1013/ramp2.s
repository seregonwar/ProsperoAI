# ramp2.s - PAI-M0 experiment G56 (wave-parallel float ramp, s_load-fed)
#
# Same kernel as G55 but k/base are read from a GPU-memory header via
# s_load_dword instead of user data. Proves the scalar-read data path
# works inside a 32-thread wave-parallel kernel (G55 had no s_load) -
# this is the x-side of a wave-parallel GEMV (x[k] is uniform across
# rows and readable with scalar loads).
#
#   c[i] = base + k * i     (i = lane id, f32)
#
# User data ABI (RSRC2 0x0C, same as G33/G35/G55):
#   s2:s3 = header (GPU mem): [0] = k (float bits), [1] = base (float)
#   s4:s5 = C (64-bit GPU VA)
# Free SGPRs s16+, VGPRs: v1 acc, v6 = saved tid.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj ramp2.s -o ramp2.o
#   llvm-objcopy --dump-section .text=ramp2.bin ramp2.o

.text
.globl ramp2

ramp2:
    s_load_dword s16, s[2:3], 0
    s_load_dword s17, s[2:3], 4
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v6, v0
    v_cvt_f32_i32 v1, v0
    v_mul_f32_e64 v1, v1, s16
    v_add_f32_e64 v1, v1, s17
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm
