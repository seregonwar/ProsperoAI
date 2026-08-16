# hbatch2.s — PAI-M0 experiments H7-H9 (load result register probes)
#
# H7: buffer_load_dword v1 -> v0 = v1 -> x4 store
# H8: buffer_load_dword v2 -> v0 = v2 -> x4 store
# H9: buffer_load_dword glc v1 -> v0 = v1 -> x4 store
# T# = 0x00080688 (executes without faulting). A in s[0:3], C in s4:s5.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj hbatch2.s -o hbatch2.o
#   llvm-objcopy --dump-section .text=hbatch2.bin hbatch2.o

.text
.globl h7
.globl h8
.globl h9

h7:
    v_mov_b32 v6, v0
    v_lshlrev_b32 v1, 2, v6
    buffer_load_dword v1, v1, s[0:3], 0 offen
    s_waitcnt vmcnt(0)
    v_mov_b32 v0, v1
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 4, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_store_dwordx4 v[2:3], v[4:7]
    s_endpgm

h8:
    v_mov_b32 v6, v0
    v_lshlrev_b32 v1, 2, v6
    buffer_load_dword v2, v1, s[0:3], 0 offen
    s_waitcnt vmcnt(0)
    v_mov_b32 v0, v2
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 4, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_store_dwordx4 v[2:3], v[4:7]
    s_endpgm

h9:
    v_mov_b32 v6, v0
    v_lshlrev_b32 v1, 2, v6
    buffer_load_dword v1, v1, s[0:3], 0 offen glc
    s_waitcnt vmcnt(0)
    v_mov_b32 v0, v1
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 4, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_store_dwordx4 v[2:3], v[4:7]
    s_endpgm
