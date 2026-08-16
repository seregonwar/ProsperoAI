# hbatch.s — PAI-M0 experiments H1/H3 (the load breakthrough attempt)
#
# H1 copy:   a[tid] -> c[4*tid .. 4*tid+3] via MUBUF load into v0 +
#            x4 broadcast store. T# = OpenAGC raw buffer format.
# H3 vecadd: c[4*tid .. 4*tid+3] = a[tid] + b[tid] (int add).
# User data (RSRC2 0x10 = 8 SGPRs):
#   s0:s3 = T#(A), s4:s7 = T#(B)  [H1: s4:s5 = C]
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj hbatch.s -o hbatch.o
#   llvm-objcopy --dump-section .text=hbatch.bin hbatch.o

.text
.globl h1
.globl h3

# H1: T#(A) in s[0:3], C in s4:s5
h1:
    v_mov_b32 v6, v0
    v_lshlrev_b32 v1, 2, v6
    buffer_load_dword v0, v1, s[0:3], 0 offen
    s_waitcnt vmcnt(0)
    v_mov_b32 v2, s4
    v_mov_b32 v3, s5
    v_lshlrev_b32 v5, 4, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_store_dwordx4 v[2:3], v[4:7]
    s_endpgm

# H3: T#(A) in s[0:3], T#(B) in s[4:7], C in s8:s9 (RSRC2 0x14 = 10)
h3:
    v_mov_b32 v6, v0
    v_lshlrev_b32 v1, 2, v6
    buffer_load_dword v0, v1, s[0:3], 0 offen
    buffer_load_dword v7, v1, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    v_add_co_u32 v0, vcc_lo, v0, v7
    v_mov_b32 v2, s8
    v_mov_b32 v3, s9
    v_lshlrev_b32 v5, 4, v6
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    flat_store_dwordx4 v[2:3], v[4:7]
    s_endpgm
