# fbatch.s — PAI-M0 experiments F1-F4
#
# Hypothesis: arithmetic with dst v0 broadcasts thread-0's result.
# Fix: compute the value into v1 (normal dst), then move to v0 for the
# store. F3/F4 test e64-VOP3 and add_co as the arithmetic instruction.
# User data (RSRC2 0x08): s0 = k, s2:s3 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj fbatch.s -o fbatch.o
#   llvm-objcopy --dump-section .text=fbatch.bin fbatch.o

.text
.globl f1
.globl f2
.globl f3
.globl f4

# F1: v_add_f32 e32 with dst v1, then v0 = v1
f1:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_mov_b32 v4, s0
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_f32 v1, v1, v4
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm

# F2: same as F1 but no v0 copy — store after value in v1 (control)
f2:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_mov_b32 v4, s0
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_f32 v1, v1, v4
    flat_store_dword v[2:3], v4
    s_endpgm

# F3: e64 VOP3 add with dst v1, then v0 = v1
f3:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_mov_b32 v4, s0
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_f32_e64 v1, v1, v4
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm

# F4: v_add_co_u32 as arithmetic (integer add), dst v1, then v0 = v1
f4:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_mov_b32 v4, s0
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v1, vcc_lo, v1, v4
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm
