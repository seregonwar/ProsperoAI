# nlexp.s - PAI-M0 experiments G71/G72 (v_rsq / v_exp serial probes)
#
# Serial-per-element model (G40/G67 validated): groups_x = n, one
# element per group, TGID_X = e. All addressing in SGPRs. Each entry
# loads x[e] from the header, applies the unknown transcendental, and
# stores to C[e]. The payload logs raw outputs to DISCOVER the 9.40
# convention (AMD v_exp_f32 is 2^x on GCN, not e^x; cos/sin here are
# turns-convention x2pi — expect surprises, verify first).
#
# User data ABI (RSRC2 0x8C, G40 pattern):
#   s2:s3 = header (h): h[0] = n (u32), h[1] = pad, h[2+e] = x[e]
#   s4:s5 = C (c[e] = f(x[e]))
#   s6    = TGID_X = element e
# Free SGPRs s10-s27; VGPRs: v1 val, v2/v3 addr.

.text
.globl nlexp_rsqrt
.globl nlexp_exp

nlexp_rsqrt:
    s_lshl_b32 s16, s6, 2
    s_add_u32 s20, s2, 8
    s_addc_u32 s21, s3, 0
    s_add_u32 s20, s20, s16
    s_addc_u32 s21, s21, 0
    s_load_dword s16, s[20:21], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v1, s16
    v_rsq_f32 v1, v1
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

nlexp_exp:
    s_lshl_b32 s16, s6, 2
    s_add_u32 s20, s2, 8
    s_addc_u32 s21, s3, 0
    s_add_u32 s20, s20, s16
    s_addc_u32 s21, s21, 0
    s_load_dword s16, s[20:21], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v1, s16
    v_exp_f32 v1, v1
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

nlexp_rcp:
    s_lshl_b32 s16, s6, 2
    s_add_u32 s20, s2, 8
    s_addc_u32 s21, s3, 0
    s_add_u32 s20, s20, s16
    s_addc_u32 s21, s21, 0
    s_load_dword s16, s[20:21], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v1, s16
    v_rcp_f32 v1, v1
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

nlexp_max:
    s_lshl_b32 s16, s6, 2
    s_add_u32 s20, s2, 8
    s_addc_u32 s21, s3, 0
    s_add_u32 s20, s20, s16
    s_addc_u32 s21, s21, 0
    s_load_dword s16, s[20:21], 0
    s_load_dword s17, s[2:3], 0
    s_waitcnt lgkmcnt(0)
    s_mul_i32 s17, s17, 4
    s_lshl_b32 s22, s6, 2
    s_add_u32 s22, s22, s17
    s_addc_u32 s23, s3, 0
    s_add_u32 s22, s2, s22
    s_addc_u32 s23, s23, 0
    s_add_u32 s22, s22, 8
    s_addc_u32 s23, s23, 0
    s_load_dword s24, s[22:23], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v1, s16
    v_mov_b32 v2, s24
    v_max_f32 v1, v1, v2
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

nlexp_min:
    s_lshl_b32 s16, s6, 2
    s_add_u32 s20, s2, 8
    s_addc_u32 s21, s3, 0
    s_add_u32 s20, s20, s16
    s_addc_u32 s21, s21, 0
    s_load_dword s16, s[20:21], 0
    s_load_dword s17, s[2:3], 0
    s_waitcnt lgkmcnt(0)
    s_mul_i32 s17, s17, 4
    s_lshl_b32 s22, s6, 2
    s_add_u32 s22, s22, s17
    s_addc_u32 s23, s3, 0
    s_add_u32 s22, s2, s22
    s_addc_u32 s23, s23, 0
    s_add_u32 s22, s22, 8
    s_addc_u32 s23, s23, 0
    s_load_dword s24, s[22:23], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v1, s16
    v_mov_b32 v2, s24
    v_min_f32 v1, v1, v2
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
