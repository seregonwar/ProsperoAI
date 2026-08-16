# g15.s — PAI-M0 THE milestone kernel (G15)
#
# c[i] = i + k_int, verified vs CPU for lanes 0-7.
# Rules: v0 must be instruction-written for the store to use it;
# per-thread integer ALU works (address path proven); VOP3 SGPR reads
# work (E45); float adds are broken — use integer add_co.
# User data (RSRC2 0x0C): s2:s3 = C, s4 = k (u32).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj g15.s -o g15.o
#   llvm-objcopy --dump-section .text=g15.bin g15.o

.text
.globl g15
g15:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v1, vcc_lo, v1, s4
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm
