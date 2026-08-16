# add1d.s - PAI-M0 north star: C[i] = A[i] + B[i]
#
# Generalizes the G22/G24 working config to a parametric 1D add:
#   - same user-data ABI as G22 (s2:s3 load base, s4:s5 store base)
#   - same G15/G22 store formula (data v4, vaddr v[2:3])
#   - s_load with IMMEDIATE offset 0 (the G24-proven form)
#   - per-element base is shifted in FREE SGPRs (s16+), not via an
#     s_load SGPR-offset and not by writing user-range SGPRs
#   - TGID_X (s6 when USER_SGPR=6 and TGID_X_EN) selects element i
#   - NUM_THREAD_X = 1 so only lane 0 stores (no 8-lane pollution)
#
# Packed input at s2:s3: dword[2*i]=A[i], dword[2*i+1]=B[i].
# C at s4:s5. RSRC2 = 0x8C (6 user SGPRs + TGID_X_EN).
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj add1d.s -o add1d.o
#   llvm-objcopy --dump-section .text=add1d.bin add1d.o

.text
.globl add1d

add1d:
    s_lshl_b32 s16, s6, 3
    s_add_u32 s18, s2, s16
    s_addc_u32 s19, s3, 0
    s_load_dword s20, s[18:19], 0
    s_load_dword s21, s[18:19], 4
    s_waitcnt lgkmcnt(0)
    s_add_u32 s20, s20, s21
    s_lshl_b32 s16, s6, 2
    s_add_u32 s18, s4, s16
    s_addc_u32 s19, s5, 0
    v_mov_b32 v2, s18
    v_mov_b32 v3, s19
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_add_co_u32 v1, vcc_lo, v0, s20
    v_mov_b32 v0, v1
    flat_store_dword v[2:3], v4
    s_endpgm
