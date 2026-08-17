# ropegen.s - PAI-M0 experiment G67/G68 (on-GPU RoPE cos/sin tables)
#
# Produces the EXACT cos_t/sin_t tables the Phase-2 decoder consumes
# (B's pai_ref_rope_cossin_f32 oracle). Serial-per-element model
# (T4/G40 validated): groups_x = ctx*r2, one element per group,
# TGID_X = e = p*r2+g. All addressing in SGPRs (s_load/s_mul/s_add),
# the only per-element value path is the G39-proven two-SGPR float mul
# + v_cos/v_sin (turns convention: x2pi on the operand).
#
# theta[p][i] = p * inv_freq[i], inv_freq[i] = base^(-i/r2).
# Host precomputes theta_turns[e] = theta/(2*pi) (so the x2pi turns
# convention cancels) and the kernel emits cos(theta)/sin(theta)
# exactly. Covers ALL rows 0..ctx-1 (no 4l+3 lane quirk - serial).
#
# G67 = cos entry @0, G68 = sin entry (writes the ctx*r2 half).
#
# User data ABI (RSRC2 0x8C, G40 pattern):
#   s2:s3 = header (h): h[0] = r2 (u32), h[1] = ctx (u32),
#                       h[2+e] = theta_turns[e] floats
#   s4:s5 = C (cos_t at [0..ctx*r2), sin_t at [ctx*r2..2*ctx*r2))
#   s6    = TGID_X = element e
# Free SGPRs s10-s27; VGPRs: v1 val, v2/v3 addr, v5 offset.

.text
.globl ropegen_cos
.globl ropegen_sin

ropegen_cos:
    s_lshl_b32 s16, s6, 2
    s_add_u32 s20, s2, 8
    s_addc_u32 s21, s3, 0
    s_add_u32 s20, s20, s16
    s_addc_u32 s21, s21, 0
    s_load_dword s16, s[20:21], 0
    s_mov_b32 s17, 0x3F800000
    s_waitcnt lgkmcnt(0)
    v_mul_f32_e64 v1, s16, s17
    v_cos_f32 v1, v1
    s_lshl_b32 s18, s6, 2
    s_add_u32 s18, s4, s18
    s_addc_u32 s19, s5, 0
    v_mov_b32 v2, s18
    v_mov_b32 v3, s19
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm

ropegen_sin:
    s_load_dword s10, s[2:3], 0
    s_load_dword s11, s[2:3], 4
    s_waitcnt lgkmcnt(0)
    s_mul_i32 s12, s10, s11
    s_lshl_b32 s12, s12, 2
    s_lshl_b32 s16, s6, 2
    s_add_u32 s20, s2, 8
    s_addc_u32 s21, s3, 0
    s_add_u32 s20, s20, s16
    s_addc_u32 s21, s21, 0
    s_load_dword s16, s[20:21], 0
    s_mov_b32 s17, 0x3F800000
    s_waitcnt lgkmcnt(0)
    v_mul_f32_e64 v1, s16, s17
    v_sin_f32 v1, v1
    s_add_u32 s18, s4, s12
    s_addc_u32 s19, s5, 0
    s_lshl_b32 s16, s6, 2
    s_add_u32 s18, s18, s16
    s_addc_u32 s19, s19, 0
    v_mov_b32 v2, s18
    v_mov_b32 v3, s19
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm
