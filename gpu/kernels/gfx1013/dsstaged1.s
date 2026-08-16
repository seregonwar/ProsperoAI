# dsstaged1.s - PAI-M0 experiment G27
#
# ONE single-dword s_load + ONE ds_write + ONE ds_read + the G15
# formula. The single s_load is G22-proven; if this completes, the
# x16+ds combo was the broken piece and we stage via 16 single loads.
# User data: s2:s3 = C (load), s4:s5 = C (store).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj dsstaged1.s -o dsstaged1.o
#   llvm-objcopy --dump-section .text=dsstaged1.bin dsstaged1.o

.text
.globl dsstaged1

dsstaged1:
    s_load_dword s10, s[2:3], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v1, 0x0
    v_mov_b32 v2, s10
    ds_write_b32 v1, v2
    v_mov_b32 v4, v0
    v_lshlrev_b32 v4, 2, v4
    ds_read_b32 v6, v4
    s_waitcnt lgkmcnt(0)
    v_readfirstlane_b32 s12, v6
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v1, vcc_lo, v0, s12
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm
