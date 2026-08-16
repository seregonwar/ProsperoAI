# PS5 GPU (9.40) — Empirical silicon rules from PAI-M0 bring-up

Every rule below was verified on physical hardware (FW 9.40) through the
autonomous deploy loop (toolchain/deploy-test.ps1). Do NOT trust
OpenAGC's documentation blindly: several claims (bit15 semantics, user
data ABI, "hardware-qualified" memset) did not hold on 9.40.

## Submission (proven working)

- `/dev/gc` open + CONTEXT_QUERY ioctl (0xC004812E): caps = 0x00000000
  on 9.40. The SPRX-confirmed recovery step (mmap GPU register space at
  0xFE0200000, PROT 0x22, MAP_SHARED on the gc fd) runs but caps stay 0.
- SUBMIT_16 ioctl (0xC0108102): queue_type = 3 works (0 also tried;
  3 is the one that executed).
- CB descriptor {0xC0023F00 header | ib_lo<<32, ib_size<<32 | ib_hi}
  + 16-dword NOP trailer = correct on 9.40.
- IT_DMA_DATA (spoofer layout) executes and verifies.
- **Fence**: action-based IT_RELEASE_MEM EOP (OpenAGC runtime layout,
  event 0x14, index 5, gcr 0x703, cache 3, data_sel 1, + 2-dword NOP)
  fires. The SetEopFlip legacy layout does NOT.
- gc reopen between experiments (pai_gpu_reset) works — a wedged ring
  does not poison the next experiment.

## Compute dispatch (empirical)

- SET_SH_REG (compute bank bit |1) + DISPATCH_DIRECT (|1, initiator
  0x41) launches waves. The full OpenAGC preamble (CONTEXT_CONTROL
  0x80000000, RESOURCE_LIMITS, dest enables, START_X/Y/Z) is NOT
  required on 9.40.
- RSRC1 = 0x602C0000 (WGP_MODE + W32_EN) executes; RSRC2 counts from
  4 to 9 user SGPRs all dispatch.
- **v0 = per-thread id** (0..31 in a wave; verified with 32-thread
  groups writing 32 distinct addresses/values). The "v9 = tid" theory
  from the OpenAGC kernel disassembly was WRONG (v9 = 0 uniformly).
- Wave size = 32 (W32 mode) with WGP_MODE set.

## FLAT memory ops (the hard-won rules)

1. **vaddr pair must be 1 (v[2:3])**. flat ops with vaddr pair 2
   (v[4:5]) HANG the wave (E1-E7, E12/E13, E19-E21 all hung with
   pair 2; G3/G5 mutation bisect isolated word1 bits [2:1]: pair 2 =
   hang, pair 1 = runs). Data register field is irrelevant.
2. **flat_store_dwordx4 broadcasts v0** ({v0,v0,v0,v0}), ignoring the
   encoded data registers (E32: constant in v0 -> 4 copies stored).
   This is why OpenAGC's "memset" writes its offset ramp: its v0 held
   the computed offset. On 9.40 that kernel CANNOT write a real
   pattern — OpenAGC's memset claim is broken on this firmware.
3. **flat_store_dword (op 0x70) with pair 1 works** as a 4-byte store
   of v0 (E36: 32 correct per-thread 4-byte writes). (Early failures
   of op 0x70 were the pair-2 issue, not the opcode.)
4. **bit15 of FLAT word0 is irrelevant** (G2: golden kernel with bit15
   cleared still executes and writes).
5. **flat_load_dword HANGS on 9.40** — every variant tried (dst v0/v1/
   v6, pair 1) hung. The golden kernel uses SMEM loads, not flat loads.
   MUBUF loads (buffer_load_dword) dispatch but the T# format is still
   unresolved (loaded zeros with the first candidate).

## Value-path arithmetic (RESOLVED 2026-08-16)

Final rules, all reproduced across F-batch experiments:

1. **s0-s1 are hardware-zeroed** ring-offset slots (the real reason the
   psbc ABI reserves them). User data must not rely on s0/s1.
2. **SGPR reads work reliably only through instructions with dst v0**
   (VOP3 form). VOP1/e64 moves into v4+ read 0. VOP1 reads of s2/s3
   (address bases) DID work — treat SGPR reads as fragile; prefer
   dst-v0 VOP3 for user data.
3. **ALU with dst v0 broadcasts thread-0's result** to all lanes
   (E42/E45: value = k + thread0(0) = k everywhere). Harmless for
   uniform scalars — use it to extract uniforms (k).
4. **Per-thread arithmetic must use dst != v0** (v1 works), with the
   result copied to v0 for the store (v_mov v0, v1).
5. **Observed store semantics (flat_store_dword, vaddr v[2:3])**:
   lanes 0..7 write (tid<<2)+3; lanes 8..31 do NOT write (exec-mask
   quirk, only 8 active lanes in WGP/W32 mode on this firmware).
   F2 verified: c[i] = 4i+3 for i<8, deterministic, CPU-checked.
6. F6/F8 (k at s4, per-thread float add): c = 0 — the value path still
   zeroes per-thread operands; combined with (3) the uniform-k
   extraction needs dst-v0, then per-thread add (dst v1) zeroed.
   The F2 formula check is the milestone evidence; the full
   c[i]=i+k kernel needs the per-thread ALU zeroing resolved.

## Loads (CLOSED DIAGNOSIS - G16/G17, 2026-08-16)

The loads do NOT return 0: every flat_load HANGS the wave AND the
ring (the fence label never fires, all subsequent submits on that
queue never execute = the M0-B FAIL chain). Proven by:

- G16 (self-ref): store 0x12345678 to c[tid] - visible on the CPU.
  load back the same address - the second store to c[tid+32] never
  ran and the EOP label never fired. c[32..35] = the 0xCC fill.
- G17 (acqrb-VA load): flat_load from the kernel own acqrb VA
  (0x200F18000) - ALSO hangs. So the read path is broken for the
  whole shader VM of our raw SUBMIT context, not just our dmem pages.
- E31 (A->C copy) = no execution observed in every run = the same
  hang, never a zero result. The historical loads-return-0 was the
  pre-filled buffers being misread.

Consistent with: stores are posted (never fault), loads block on the
VM fault/return that never completes for the raw /dev/gc context on
9.40. The kernel own GPU mappings (acqrb) hang too => the shader
READ capability of our context is the missing piece, not the page
tables.

## Reads - the three paths (G16-G22, 2026-08-16)

Three distinct read behaviors, all on the same buffers:

- flat_load_dword: HANGS the wave AND the ring (G16/G17/G18, also
  without any s_waitcnt). The EOP label never fires. Do not use.
- buffer_load_dword (MUBUF): completes (label fires) but returns 0
  for ALL T# word3 candidates tried (G21 matrix: 0x31014FAC, 0x00080000,
  0x00080001, 0x80000400, 0x20002000 - identical 4i+3 output). The
  vector read path zero-fills; the T# format is NOT the discriminator.
  with T# word3 = 0x31014FAC (G20/G21: stored 4i+0+3). The T# format
  for 9.40 is still unresolved; the vector path itself does not hang.
- s_load_dword (SMEM): WORKS. G22 PASS: s_load of C[0] returned the
  real 0xA5A5A5A5 fill; the G15 store formula produced
  c[i] = 4i + 0xA5A5A5A5 + 3 exactly. The scalar read path is fully
  functional on 9.40.

Consequences: uniform/shader-constant data is readable via s_load.
Per-thread reads need either the correct MUBUF T# word3 (hunt still
open: 0x31014FAC and 0x20002000 both return 0) or an SMEM-based
restructuring (s_load_dwordx16 + LDS redistribution).

Also ruled out for the hang/zeros (safe tests): dmem type 1 vs 3
(WB_ONION vs WC_GARLIC + clflush), the exact GPU authid
0x4801000000000000, MAKESYSMAP ioctl (0xC0088109, returns identity),
SETUP_ASYNC ioctl (0x80048126, rc=0). None changed the read behavior.

## Loads (OPEN — the next hard problem)

All vector/scalar loads return 0 on 9.40 in our dispatch config:
- flat_load_dword: HANGS (all variants: dst v0/v1/v6, pair 1, glc)
- buffer_load_dword (MUBUF): executes but returns 0 — with T#
  0x31014FAC (OpenAGC raw) and 0x20002000 the load FAULTS the wave;
  with 0x00080688 and 0x97688 it returns 0 regardless of the vdata
  register (v0/v1/v2) and glc.
- s_load_dwordx4 (SMEM, the golden's path): returns 0 — even with the
  proven sbase pair s[2:3] and with the destination at s[4:7] (H10,
  H11, H12).

Falsified hypotheses (all tested on hardware):
- s0-s1 zeroing eating load destinations/T# bases (H12: load into
  s[4:7] still 0)
- L1 cache staleness (IT_ACQUIRE_MEM GCR_ALL=0xC3B1 before loads: no
  change)
- dmem vs flexible-memory GPU mappings (both return 0)
- missing driver context: QUEUE_CREATE with SPRX magic tokens
  SUCCEEDS on 9.40 (acqrb/eop regions accepted), but ACB submission
  (queue 0xc + const-IB descriptors) does NOT execute — the real
  compute-queue path needs the DingDong/ring machinery.

Stores land correctly (DMA + flat stores verified), so the write path
is fine. Remaining candidates: the wave's GPU page-table READ side
(zero-fill on fault — writes may be posted/uncached while reads fault),
or the driver's shadow/ring setup. The gvmspace array was NOT found by
structural scans of kernel_940.elf (layout differs from the 11.20
spoofer). Next: runtime gvmspace discovery (scan kernel data for an
entry CONTAINING the live acqrb/dmem VAs) and the PTE fix, or the
DingDong submission path on the created queue.

## MILESTONE STATUS (2026-08-16)

- **PAI-M0 GPU compute: COMPLETE** on FW 9.40 (G15/G22 path).
  Runtime-parameterized integer arithmetic, scalar `s_load`, and
  `flat_store` writeback are CPU-reference-checked.
- Final store semantics (empirical, reproduced across G7-G15):
  - `flat_store_dword` data = (last instruction's SGPR-sourced value k)
    + (vaddr offset, tid*4) + 3. With an instruction-written literal in
    v0 it writes v0 verbatim (G14/E36/G9).
  - `flat_store_dwordx4` = v0 broadcast x4 (E30/E32/G9).
  - Only lanes 0-7 of a 32-thread wave write (exec-mask quirk).
  - vaddr must be v[2:3]; v[4:5] hangs.
- Float ALU: FULLY UNLOCKED (G35/G39 PASS). SALU float does NOT exist on gfx1013 (llvm-mc: no s_mul_f32/s_add_f32) - float is VALU-only. Validated: v_cvt_f32_i32 (G28), v_add_f32_e64 with direct SGPR scalar (G35), v_mul_f32_e64 with both-SGPR operands + VGPR+VGPR add accumulator (G39 float dot, exact match vs CPU). Forms to avoid: dst-v0 float op = HANG, e32 VOP2 mixed VGPR+SGPR = VGPR zeroed, v_mov_b32 vX, s4 (dst != v0) = SGPR read 0.
- **PAI-M1 (serial G22 tensor primitives): CLOSED without LDS**
  - M1A integer SAXPY: VALIDATED (N=8..1M, stable multi-iter)
  - M1B `dot_serial_u32`: VALIDATED (correctness reduction; not fast)
  - M1C parallel reduction: **BLOCKED** on LDS (see below)
  - M1D serial-per-row GEMV: VALIDATED through 256×1024; submit→EOP
    apparent bandwidth ~2–4 GB/s (not HBM)
- **LDS / M1C (STOP hunting RSRC2):** G25 FAIL even with OpenAGC
  gfx1013 minimum 1 KiB (`RSRC2=0x1000C`, `LDS_SIZE=2`), `m0=0`, and
  store of LDS result via v0. Next unlock = real Shader CS AGC blob
  that uses LDS; dump `COMPUTE_PGM_RSRC2` at SH offset `0x213`. Do not
  invent further LDS_SIZE values. OpenAGC (`_vendor/OpenAGC`,
  Apache-2.0) is useful for register/layout rules, not for an LDS
  RSRC2 dump.

## T4 serial kernel family (G42-G54, 2026-08-16)

Hardware run 004416 (console 9021): **G42-G48 float + G49-G54 integer
= 13/13 PASS**, CPU-reference-checked through the deploy loop.

Kernels (gfx1013, `t4_ops.s` / `int_ops.s`, host-ref mirrors in
`gpu/hal/host_kernels.c`):

- float G42-G46 elementwise (add/sub/mul/relu/clip) — packed (a,b)
  pairs, one group per element, VALU float e64 direct-SGPR form;
- G47 `biasadd` — header [cols, a, bias], one group per row;
- G48 `matmul` — header [K, N, a, b], one group per row;
- G49-G54 integer — `add2d`/`sub1d`/`mul1d`/`relu`/`clip`/`matmul`
  u32, same serial pattern.

New rules confirmed during T4/T4C:

1. **Loop kernels dispatch one group per cell**: biasadd/matmul with a
   single thread per group must loop over cells inside the kernel
   (group_x = rows), not rely on thread-per-element dispatch — the
   serial model walks the whole output per wave.
2. **Division-style loops need an unconditional branch-back**: the
   per-cell loop counter decrements with `s_sub` + unconditional
   `s_branch` back; a conditional exit on the counter produced
   incorrect iteration counts on silicon.
3. **Float accumulation compare is tolerance-based**: G48 matmul
   accumulates 16 terms with `v_add_f32` (serial order) against a
   host serial sum — compared with `pai_ref_compare_f32` at 1e-6
   (relative+absolute), not bit-exact.
4. **G48 b k-stride is N*4 bytes** (row-major [k][N]); an earlier
   version advanced `b` by N bytes (fix `a06439b`, `s15 = s23 << 2`).
5. Float elementwise/biasadd/matmul were validated on HW only after
   the float ALU e64 rules from G35/G39/G40/G41 (see MILESTONE
   below): direct SGPR operands, VGPR+VGPR accumulator, dst != v0.

## Toolchain

- llvm-mc 18 (Windows, ps5-payload-sdk/tools/llvm18/bin) assembles
  gfx1013; llvm-mc 14 (WSL) produces identical encodings for the
  tested subset. ACO's kernel contains instructions LLVM's gfx1013
  model does not know (v_lshl_add_u32 etc.) — LLVM's model is
  conservative, the silicon is fuller RDNA2.
- llvm-objdump 14/18 cannot decode the OpenAGC kernel bytes (they use
  encodings outside LLVM's tables).
- Deploy loop: `toolchain/deploy-test.ps1` (PS5 console online).

## Kernel RE of kernel_940.elf

- gc driver code ~VA 0x72F000-0x739000; ioctl dispatcher found
  (dir switch at +0x1977, jump table +0x1a8c); the kernel parses CBs
  and has special paths for PM4 opcodes 0xB5/0xEF/0xCA. gc_suspend/
  resume events appear in klog for system processes.
- Klog available via FTP /data/klog/klog.log.

## Status

- Working: bootstrap, jailbreak, /data logging, lifecycle listener,
  notify, DMA, fence, dispatch, SMEM loads, integer ALU, stores,
  add1d / SAXPY / serial dot / serial GEMV on the G22 path; float ALU
  (G35/G39/G40/G41); T4 serial kernel family G42-G54 float + integer
  (13/13, run 004416).
- Open (do not block M1 closeout): MUBUF T# / flat loads, lane 8+
  exec mask, LDS (M1C) pending AGC CS blob @ 0x213.


