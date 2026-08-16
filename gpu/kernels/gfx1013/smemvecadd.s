# smemvecadd.s - PAI-M0 experiment G23
#
# The first SMEM->LDS staging compute: 32 elements per dispatch.
#   s_load_dwordx16 A[off..off+15] and B[off..off+15] (scalar path)
#   idempotent LDS writeback (every lane writes the same values)
#   barrier, per-lane ds_read of a[i] and b[i], integer add
#   store c[i] with the G16-exact v0-operand pattern
# User data: s2:s3 = A, s4:s5 = B, s6:s7 = C, s8 = element offset.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj smemvecadd.s -o smemvecadd.o
#   llvm-objcopy --dump-section .text=smemvecadd.bin smemvecadd.o

.text
.globl smemvecadd

smemvecadd:
    s_lshl_b32 s9, s8, 2
    s_load_dwordx16 s[16:31], s[2:3], s9
    s_load_dwordx16 s[32:47], s[4:5], s9
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v1, 0x0
    v_mov_b32 v2, s10
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x4
    v_mov_b32 v2, s11
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x8
    v_mov_b32 v2, s12
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0xc
    v_mov_b32 v2, s13
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x10
    v_mov_b32 v2, s14
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x14
    v_mov_b32 v2, s15
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x18
    v_mov_b32 v2, s16
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x1c
    v_mov_b32 v2, s17
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x20
    v_mov_b32 v2, s18
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x24
    v_mov_b32 v2, s19
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x28
    v_mov_b32 v2, s20
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x2c
    v_mov_b32 v2, s21
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x30
    v_mov_b32 v2, s22
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x34
    v_mov_b32 v2, s23
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x38
    v_mov_b32 v2, s24
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x3c
    v_mov_b32 v2, s25
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x40
    v_mov_b32 v2, s26
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x44
    v_mov_b32 v2, s27
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x48
    v_mov_b32 v2, s28
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x4c
    v_mov_b32 v2, s29
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x50
    v_mov_b32 v2, s30
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x54
    v_mov_b32 v2, s31
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x58
    v_mov_b32 v2, s32
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x5c
    v_mov_b32 v2, s33
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x60
    v_mov_b32 v2, s34
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x64
    v_mov_b32 v2, s35
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x68
    v_mov_b32 v2, s36
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x6c
    v_mov_b32 v2, s37
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x70
    v_mov_b32 v2, s38
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x74
    v_mov_b32 v2, s39
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x78
    v_mov_b32 v2, s40
    ds_write_b32 v1, v2
    v_mov_b32 v1, 0x7c
    v_mov_b32 v2, s41
    ds_write_b32 v1, v2
    s_barrier
    v_mov_b32 v4, v0
    v_lshlrev_b32 v4, 2, v4
    ds_read_b32 v6, v4
    v_add_co_u32 v4, vcc_lo, v4, 64
    ds_read_b32 v7, v4
    s_waitcnt lgkmcnt(0)
    v_add_co_u32 v1, vcc_lo, v6, v7
    v_mov_b32 v2, s6
    v_mov_b32 v3, s7
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm
