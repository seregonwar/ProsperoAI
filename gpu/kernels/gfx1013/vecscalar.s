# vecscalar.s — PAI-M0 THE milestone kernel (F6)
#
# c[i] = (float)i + k, verified against the CPU reference.
# Rules applied:
#   - s0-s1 are hardware-zeroed ring-offset slots: k lives at s4
#   - arithmetic dst must not be v0 (broadcast): compute in v1
#   - flat store: vaddr pair v[2:3], data from v0
# User data (RSRC2 0x0C): s2:s3 = C, s4 = k (float).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj vecscalar.s -o vecscalar.o
#   llvm-objcopy --dump-section .text=vecscalar.bin vecscalar.o

.text
.globl vecscalar
vecscalar:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_mov_b32_e64 v4, s4
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_f32 v1, v1, v4
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm
