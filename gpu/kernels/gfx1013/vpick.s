# vpick.s - PAI-M0 experiment G59 (direct v16 read, no movrels)
#
# Decisive bisection for G57: is v_movrels_b32 broken on 9.40, or do
# the v_mov_b32 v16..v31 block copies not land (RSRC1 VGPRS)?
#
#   header[0..15] = 16 values
#   c[lane] = v16 (direct read, ALL lanes) at C + lane*4
#
# If c[0..7] == header[0] (0x10000000 everywhere): v16 IS populated
#   -> v_movrels_b32 is the non-functional piece (documented negative).
# If c == garbage: the v16+ copies are dropped -> RSRC1 VGPRS wrong.
#
# User data ABI (RSRC2 0x0C): s2:s3 = header, s4:s5 = C.

.text
.globl vpick

vpick:
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
    v_mov_b32 v6, v0        /* save tid */
    v_mov_b32 v1, v16       /* direct read - no v_movrels */
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm
