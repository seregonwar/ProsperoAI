# fbatch3.s - PAI-M0 experiments G31/G32 (float add operand probes)
#
# G31: cvt + v_add_f32_e64 with dst v0 (the broadcast rule probe)
# G32: cvt + v_add_f32 of two VGPRs (v1+v2, dst v1)
# v6 keeps the original tid for the address math.
# User data (RSRC2 0x0C): s2:s3 = C, s4 = k (float).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj fbatch3.s -o fbatch3.o
#   llvm-objcopy --dump-section .text=fbatch3.bin fbatch3.o

.text
.globl g31
.globl g32

g31:
    v_mov_b32 v6, v0
    v_cvt_f32_i32 v1, v0
    v_add_f32_e64 v0, v1, s4
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_store_dword v[2:3], v0
    s_endpgm

g32:
    v_mov_b32 v6, v0
    v_cvt_f32_i32 v1, v0
    v_cvt_f32_i32 v2, v0
    v_add_f32_e64 v1, v1, v2
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm
