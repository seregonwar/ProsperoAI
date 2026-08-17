# lanepick.s - PAI-M0 experiment G57 (per-lane select from s_load block)
#
# Probe for the wave-parallel GEMV unlock: can a lane pick its own
# element out of a 16-dword block loaded with s_load_dwordx16?
#
#   header[0..15] = 16 values (0..15 pattern)
#   c[lane] = header[lane] for the 8 storing lanes
#
# Mechanism: s_load_dwordx16 loads the block into s[16:31] (uniform),
# copy to v[16:31] (per-lane copies), then v_movrels_b32 reads
# vreg[m0_base + lane] - VGPR-relative read with the per-lane index in
# the source operand (no cross-lane comm, no barrier, no vector load).
# m0 = base VGPR index in dword units (G58 proved the block lands in
# s[16:31]; m0=64 leaked the source, m0=16 addresses v16..v23).
#
# User data ABI (RSRC2 0x0C): s2:s3 = header, s4:s5 = C.
# Free SGPRs s16+, VGPRs v16-v31 (block copies), v6 = saved tid.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj lanepick.s -o lanepick.o
#   llvm-objcopy --dump-section .text=lanepick.bin lanepick.o

.text
.globl lanepick

lanepick:
    s_load_dwordx16 s[16:31], s[2:3], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v16, s16
    v_mov_b32 v17, s17
    v_mov_b32 v18, s18
    v_mov_b32 v19, s19
    v_mov_b32 v20, s20
    v_mov_b32 v21, s21
    v_mov_b32 v22, s22
    v_mov_b32 v23, s23
    v_mov_b32 v24, s24
    v_mov_b32 v25, s25
    v_mov_b32 v26, s26
    v_mov_b32 v27, s27
    v_mov_b32 v28, s28
    v_mov_b32 v29, s29
    v_mov_b32 v30, s30
    v_mov_b32 v31, s31
    s_mov_b32 m0, 16
    v_mov_b32 v6, v0
    v_movrels_b32 v1, v0
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm
