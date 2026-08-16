# hbatch5.s — PAI-M0 experiments H12/H13 (the zeroed-s0-s1 workaround)
#
# s0-s1 are hardware-zeroed ring-offset slots: anything there (load
# destinations, T# bases) reads as 0. Move everything to s2+.
#
# H12: s_load_dwordx4 s[4:7], s[2:3], 0 -> s4 = A[0]; store trick writes
#      (last SGPR s4) + offset + 3 in place on A.
#      Expected: A[i] = A[0]_orig + 4i + 3.
# H13: T#(A) at s[4:7], C at s2:s3; buffer_load_dword v0 -> x4 store.
#      Expected: c[4i..4i+3] = a[i].
# User data: H12 RSRC2 0x0C (s2:s3=A), H13 RSRC2 0x10 (s2:s3=C, s4:s7=T#).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj hbatch5.s -o hbatch5.o
#   llvm-objcopy --dump-section .text=hbatch5.bin hbatch5.o

.text
.globl h12
.globl h13

h12:
    v_mov_b32 v1, v0
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    s_load_dwordx4 s[4:7], s[2:3], 0
    s_waitcnt lgkmcnt(0)
    v_add_co_u32 v6, vcc_lo, v5, s4
    flat_store_dword v[2:3], v4
    s_endpgm

h13:
    v_mov_b32 v6, v0
    v_lshlrev_b32 v1, 2, v6
    buffer_load_dword v0, v1, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 4, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_store_dwordx4 v[2:3], v[4:7]
    s_endpgm
