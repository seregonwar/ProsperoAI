# store64_v0.s — PAI-M0 experiment E32
#
# Hypothesis from E30 behavior: on 9.40 the flat_store_dwordx4
# broadcasts v0 as the data (4 dwords), ignoring the encoded data
# registers. Kernel: compute per-thread address in v[2:3], then move
# the constant into v0 right before the store.
# User data: s0-s1 ring offsets, s2:s3 = dst.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj store64_v0.s -o store64_v0.o
#   llvm-objcopy --dump-section .text=store64_v0.bin store64_v0.o

.text
.globl store64_v0
store64_v0:
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_mov_b32 v1, v0
    v_lshlrev_b32 v1, 4, v1
    v_add_co_u32 v2, vcc_lo, v2, v1
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, 0x12345678
    flat_store_dwordx4 v[2:3], v[4:7]
    s_endpgm
