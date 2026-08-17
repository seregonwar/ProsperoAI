# movrels.s - PAI-M0 experiment G64 (v_movrels_b32 with in-ceiling block)
#
# The one untested piece: v_movrels_b32 itself. G57's copies went to
# v16..v31 which the 16-VGPR hardware ceiling drops (G59/G63 proved
# v16+ reads return tid), so movrels read garbage there. Here the
# s_load block is copied to v7..v14 (within the proven v0..v15 range,
# G62) and m0=7 makes v_movrels read v[7+tid]:
#
#   header[0..15] = 16 values
#   c[lane] = header[7 + lane]   (lanes 0..7 store)
#
# If PASS: per-lane selection works -> wave-parallel GEMV buildable.
# If FAIL (tid leak): v_movrels itself is broken on 9.40 -> fallback
# to DS roundtrip or arithmetic per-lane value path.
#
# Register plan (all within v0..v15 ceiling):
#   v0  = tid (movrels index)
#   v1  = selected value (result)
#   v2/v3 = C address, v5 = offset, v6 = saved tid
#   v7..v14 = block values s16..s23
#
# User data ABI (RSRC2 0x0C): s2:s3 = header, s4:s5 = C.

.text
.globl movrels

movrels:
    s_load_dwordx16 s[16:31], s[2:3], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v7, s16
    v_mov_b32 v8, s17
    v_mov_b32 v9, s18
    v_mov_b32 v10, s19
    v_mov_b32 v11, s20
    v_mov_b32 v12, s21
    v_mov_b32 v13, s22
    v_mov_b32 v14, s23
    s_mov_b32 m0, 7
    v_mov_b32 v6, v0        /* save tid */
    v_movrels_b32 v1, v0    /* v1 = v[7 + tid] */
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm
