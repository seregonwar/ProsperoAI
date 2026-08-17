# blockdump.s - PAI-M0 experiment G58 (s_load_dwordx16 block dump)
#
# Bisection probe for G57: does s_load_dwordx16 populate s[16:31] at
# all when 32 SGPRs are allocated (RSRC1 0x602C0043)?
#
#   header[0..15] = 16 values
#   c[0..7] = s16..s23 via direct v_mov copies (no v_movrels, no m0)
#
# If c == header[0..7], the block SMEM load works and G57's failure
# is in v_movrels/m0. If c == 0, s[16:31] never populated.
#
# User data ABI (RSRC2 0x0C): s2:s3 = header, s4:s5 = C.

.text
.globl blockdump

blockdump:
    s_load_dwordx16 s[16:31], s[2:3], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v0, s16
    v_mov_b32 v1, s17
    v_mov_b32 v2, s18
    v_mov_b32 v3, s19
    v_mov_b32 v4, s20
    v_mov_b32 v5, s21
    v_mov_b32 v6, s22
    v_mov_b32 v7, s23
    v_mov_b32 v8, s4   /* C low */
    v_mov_b32 v9, s5   /* C high */
    s_mov_b32 m0, 0
    flat_store_dword v[8:9], v0   /* c[0] = s16 */
    v_add_co_u32 v8, vcc_lo, v8, 4
    v_add_co_ci_u32 v9, vcc_lo, v9, 0, vcc_lo
    flat_store_dword v[8:9], v1   /* c[1] = s17 */
    v_add_co_u32 v8, vcc_lo, v8, 4
    v_add_co_ci_u32 v9, vcc_lo, v9, 0, vcc_lo
    flat_store_dword v[8:9], v2   /* c[2] = s18 */
    v_add_co_u32 v8, vcc_lo, v8, 4
    v_add_co_ci_u32 v9, vcc_lo, v9, 0, vcc_lo
    flat_store_dword v[8:9], v3   /* c[3] = s19 */
    v_add_co_u32 v8, vcc_lo, v8, 4
    v_add_co_ci_u32 v9, vcc_lo, v9, 0, vcc_lo
    flat_store_dword v[8:9], v4   /* c[4] = s20 */
    v_add_co_u32 v8, vcc_lo, v8, 4
    v_add_co_ci_u32 v9, vcc_lo, v9, 0, vcc_lo
    flat_store_dword v[8:9], v5   /* c[5] = s21 */
    v_add_co_u32 v8, vcc_lo, v8, 4
    v_add_co_ci_u32 v9, vcc_lo, v9, 0, vcc_lo
    flat_store_dword v[8:9], v6   /* c[6] = s22 */
    v_add_co_u32 v8, vcc_lo, v8, 4
    v_add_co_ci_u32 v9, vcc_lo, v9, 0, vcc_lo
    flat_store_dword v[8:9], v7   /* c[7] = s23 */
    s_endpgm
