# fdot_serial.s - PAI-M0 experiment G39 (VALU float dot, serial)
#
# Float32 dot via the serial SMEM path + the unlocked VALU float form:
#   v_mul_f32_e64 v2, s20, s21    (both SGPRs direct - G35 rule)
#   v_add_f32_e64 v1, v1, v2      (VGPR+VGPR - G32 rule)
# The accumulator v1 is a VGPR, loop-carried. One thread, one group.
#
#   sum = 0.0f
#   for i = 0 .. N-1: sum += a[i] * b[i]
#   store sum            # C[0]
#
# Packed header at s2:s3 (imm 0 = N), (a,b) interleaved, C at s4:s5.
# RSRC2 = 0x8C (the dot_serial_u32 config). Free SGPRs s17-s19.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj \
#       fdot_serial.s -o fdot_serial.o
#   llvm-objcopy --dump-section .text=fdot_serial.bin fdot_serial.o

.text
.globl fdot_serial

fdot_serial:
    s_load_dword s17, s[2:3], 0
    s_add_u32 s18, s2, 8
    s_addc_u32 s19, s3, 0
    v_mov_b32 v1, 0
    s_waitcnt lgkmcnt(0)
    s_cmp_eq_u32 s17, 0
    s_cbranch_scc1 .Ldone

.Lloop:
    s_load_dword s20, s[18:19], 0
    s_load_dword s21, s[18:19], 4
    s_waitcnt lgkmcnt(0)
    v_mul_f32_e64 v2, s20, s21
    v_add_f32_e64 v1, v1, v2
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
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v0
    s_endpgm