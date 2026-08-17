# vpick2.s - PAI-M0 experiment G60 (v8..v15 block, direct read)
#
# Continuation of G59: v16+ direct reads returned tid. Golden kernel
# (HW-qualified) uses v6-v9 fine with STANDARD RSRC1 0x602C0000
# (VGPRS=0). Test mid-range v8..v15 under the SAME standard RSRC1:
# if v15 works, the ceiling is >= 16 under the golden config and the
# v16+ failure is real (or BLOCK32 RSRC1 is what broke it).
#
#   header[0..15] = 16 values
#   c[lane] = v8 (direct read) at C + lane*4
#
# If c[0..7] == header[0]: v8+ copies land under standard RSRC1.
# If c == tid: v8+ broken even under golden config -> ceiling < 9
#   contradicts golden -> different mechanism.
#
# User data ABI (RSRC2 0x0C): s2:s3 = header, s4:s5 = C.

.text
.globl vpick2

vpick2:
    s_load_dwordx16 s[16:31], s[2:3], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v8, s16
    v_mov_b32 v9, s17
    v_mov_b32 v10, s18
    v_mov_b32 v11, s19
    v_mov_b32 v12, s20
    v_mov_b32 v13, s21
    v_mov_b32 v14, s22
    v_mov_b32 v15, s23
    v_mov_b32 v6, v0        /* save tid */
    v_mov_b32 v1, v8        /* direct read of v8 */
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm
