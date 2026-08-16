# v0model.s — PAI-M0 experiments E34-E37 (flat v0-broadcast model)
#
# Empirically confirmed on 9.40: flat_store_dwordx4 v[2:3], _ writes
# {v0,v0,v0,v0} (data regs ignored), and only vaddr pair 1 (v[2:3])
# works. This file probes the load side and builds the real vecadd.
#
# E34 copy_v0:     flat_load_dword v0, v[2:3] then x4 store -> copy
# E35 load_v1:     load into v1, copy to v0, store -> dst!=v0 allowed?
# E36 store_dw:    flat_store_dword v[2:3] -> 1 dword broadcast of v0?
# E37 vecadd_v0:   A->v0, B->v1, v_add_f32 v0, store dword -> c=a+b
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj v0model.s -o v0model.o
#   llvm-objcopy --dump-section .text=v0model.bin v0model.o

.text
.globl copy_v0
.globl load_v1
.globl store_dw
.globl vecadd_v0

# user data: s2:s3 = A, s4:s5 = C
copy_v0:
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_mov_b32 v4, s4
    v_mov_b32 v5, s5
    v_mov_b32 v1, v0
    v_lshlrev_b32 v1, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v1
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_load_dword v0, v[2:3]
    s_waitcnt vmcnt(0) lgkmcnt(0)
    v_add_co_u32 v2, vcc_lo, v4, v1
    v_add_co_ci_u32 v3, vcc_lo, v5, 0, vcc_lo
    flat_store_dwordx4 v[2:3], v[4:7]
    s_endpgm

# user data: s2:s3 = A, s4:s5 = C
load_v1:
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_mov_b32 v4, s4
    v_mov_b32 v5, s5
    v_mov_b32 v1, v0
    v_lshlrev_b32 v1, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v1
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_load_dword v1, v[2:3]
    s_waitcnt vmcnt(0) lgkmcnt(0)
    v_mov_b32 v0, v1
    v_add_co_u32 v2, vcc_lo, v4, v1
    v_add_co_ci_u32 v3, vcc_lo, v5, 0, vcc_lo
    flat_store_dwordx4 v[2:3], v[4:7]
    s_endpgm

# user data: s2:s3 = dst
store_dw:
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_mov_b32 v1, v0
    v_lshlrev_b32 v1, 2, v1
    v_add_co_u32 v2, vcc_lo, v2, v1
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, 0xABCDDCBA
    flat_store_dword v[2:3], v4
    s_endpgm

# user data: s2:s3 = A, s4:s5 = B, s6:s7 = C
vecadd_v0:
    v_mov_b32 v4, s2
    v_mov_b32 v5, s3
    v_mov_b32 v6, s4
    v_mov_b32 v7, s5
    v_mov_b32 v8, s6
    v_mov_b32 v9, s7
    v_mov_b32 v1, v0
    v_lshlrev_b32 v1, 2, v1
    v_add_co_u32 v2, vcc_lo, v4, v1
    v_add_co_ci_u32 v3, vcc_lo, v5, 0, vcc_lo
    flat_load_dword v0, v[2:3]
    v_add_co_u32 v2, vcc_lo, v6, v1
    v_add_co_ci_u32 v3, vcc_lo, v7, 0, vcc_lo
    flat_load_dword v1, v[2:3]
    s_waitcnt vmcnt(0) lgkmcnt(0)
    v_add_f32 v0, v0, v1
    v_add_co_u32 v2, vcc_lo, v8, v1
    v_add_co_ci_u32 v3, vcc_lo, v9, 0, vcc_lo
    flat_store_dword v[2:3], v4
    s_endpgm
