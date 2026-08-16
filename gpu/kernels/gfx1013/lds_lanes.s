# lds_lanes.s - PAI-M1C probe: per-lane LDS roundtrip, no barrier
#
# NUM_THREAD_X = 8. Each lane writes and reads LDS[tid*4] so visibility
# does not need s_barrier. Vector store of the loaded dword (not the
# scalar G22 path, which is 1-lane). RSRC2 = USER_SGPR=6 | LDS_SIZE=1.
#
# User data: s2:s3 = C.
#
# Assemble:
#   llvm-mc -triple=amdgcn -mcpu=gfx1013 -filetype=obj \
#       lds_lanes.s -o lds_lanes.o
#   llvm-objcopy --dump-section .text=lds_lanes.bin lds_lanes.o

.text
.globl lds_lanes

lds_lanes:
    s_mov_b32 m0, 0
    v_lshlrev_b32 v1, 2, v0
    v_or_b32 v2, 0xA5000000, v0
    ds_write_b32 v1, v2
    s_waitcnt lgkmcnt(0)
    ds_read_b32 v6, v1
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v2, s2
    v_mov_b32 v3, s3
    v_lshlrev_b32 v5, 2, v0
    v_add_co_u32 v2, vcc_lo, v2, v5
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_mov_b32 v0, v6
    flat_store_dword v[2:3], v0
    s_endpgm
