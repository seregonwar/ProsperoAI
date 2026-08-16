# dot_serial_u32.s - PAI-M1B scalar reduction correctness primitive
#
# NOT a performance kernel. Validates: GPU loop, address increment,
# repeated s_load, mul/add dependency chain, SGPR accumulator, one
# scalar store. Parallel reduction (LDS / wave ops / atomics) is M1C.
#
#   sum = 0
#   for i = 0 .. N-1:
#       a = load A[i]
#       b = load B[i]
#       sum += a * b          # uint32 wrap
#   store sum                 # C[0]
#
# One workgroup, NUM_THREAD_X = 1, groups_x = 1. N comes from a packed
# header (s_load imm 0). Per-element pointer lives in FREE SGPRs and
# advances by 8; s_load uses IMMEDIATE offsets 0 and 4 (G22 contract).
# No LDS, no barrier, no TGID, no s_load SGPR-offset.
#
# Packed layout: dword[0]=N, dword[1]=pad, dword[2+2*i]=A[i],
# dword[3+2*i]=B[i]. User data: s2:s3 pack, s4:s5 C. RSRC2 = 0x8C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj \
#       dot_serial_u32.s -o dot_serial_u32.o
#   llvm-objcopy --dump-section .text=dot_serial_u32.bin dot_serial_u32.o

.text
.globl dot_serial_u32

dot_serial_u32:
    s_load_dword s17, s[2:3], 0
    s_add_u32 s18, s2, 8
    s_addc_u32 s19, s3, 0
    s_mov_b32 s22, s0
    s_waitcnt lgkmcnt(0)
    s_cmp_eq_u32 s17, 0
    s_cbranch_scc1 .Ldone

.Lloop:
    s_load_dword s20, s[18:19], 0
    s_load_dword s21, s[18:19], 4
    s_waitcnt lgkmcnt(0)
    s_mul_i32 s20, s20, s21
    s_add_u32 s22, s22, s20
    s_add_u32 s18, s18, 8
    s_addc_u32 s19, s19, 0
    s_sub_u32 s17, s17, 1
    s_cmp_lg_u32 s17, 0
    s_cbranch_scc1 .Lloop

.Ldone:
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v1, vcc_lo, v0, s22
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm
