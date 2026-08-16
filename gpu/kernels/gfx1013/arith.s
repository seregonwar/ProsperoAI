# arith.s — PAI-M0 experiment E42 (THE milestone kernel)
#
# c[i] = (float)i + k for i < 32 (4-byte dword store, stride 4).
# Pure per-thread GPU arithmetic, no loads.
# User data: s2:s3 = C, s4 = k (float), s5 = padding (even SGPR count).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj arith.s -o arith.o
#   llvm-objcopy --dump-section .text=arith.bin arith.o

.text
.globl arith_v0
arith_v0:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_add_f32 v0, v1, s4
    v_lshlrev_b32 v1, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v1
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_store_dword v[2:3], v4
    s_endpgm
