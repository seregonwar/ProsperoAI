# dsstaged.s - PAI-M0 experiment G26
#
# One s_load_dwordx16 + ONE ds_write + ONE ds_read + the G15 store
# formula with the readfirstlane. The minimal x16+LDS combo to bisect
# the G23 hang further.
# User data: s2:s3 = C (load), s4:s5 = C (store).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj dsstaged.s -o dsstaged.o
#   llvm-objcopy --dump-section .text=dsstaged.bin dsstaged.o

.text
.globl dsstaged

dsstaged:
    s_load_dwordx16 s[16:31], s[2:3], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v1, 0x0
    v_mov_b32 v2, s16
    ds_write_b32 v1, v2
    v_mov_b32 v4, v0
    v_lshlrev_b32 v4, 2, v4
    ds_read_b32 v6, v4
    s_waitcnt lgkmcnt(0)
    v_readfirstlane_b32 s8, v6
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v1, vcc_lo, v0, s8
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm
