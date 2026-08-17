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
- G47 `biasadd` — header [cols, a, bias], one group per cell;
- G48 `matmul` — header [K, N, a, b], one group per cell;
- G49-G54 integer — `add2d`/`sub1d`/`mul1d`/`relu`/`clip`/`matmul`
  u32, same serial pattern.

New rules confirmed during T4/T4C:

1. **Loop kernels dispatch one group per cell**: biasadd/matmul with a
   single thread per group take `group_x = rows*cols` and derive the
   (row, col) pair from the group id with a subtraction loop (i = g/N,
   j = g%N via repeated `s_sub`), not thread-per-element dispatch —
   the serial model walks the whole output per wave.
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

## G55 wave-parallel float ramp (2026-08-17, run 011141)

First kernel past the serial NUM_THREAD_X=1 model: `ramp.s` computes
`c[i] = base + k*i` with NUM_THREAD_X=32, one group, and the 8 storing
lanes of the wave. HW-validated PASS.

- User data ABI (RSRC2 0x0C, same as G33/G35): s2:s3 = C, s4 = k
  (float), s5 = base (float); free SGPRs s16+, VGPRs v1 (acc), v6
  (saved tid).
- Float form used: `v_cvt_f32_i32` (int->float, G35 rule),
  `v_mul_f32_e64` with direct SGPR operand, `v_add_f32_e64`
  VGPR+SGPR accumulator, dst != v0, final v0 copy for the store.
- **Value-path quirk confirmed again**: lane i stores
  `base + k*(4i+3)`, not `base + k*i` — the G15 store formula
  (`tid*4+3`) is baked into the VALU->store data path, exactly as
  G35's check (`(4i+3)+k`) observed. The oracle must use the same
  convention (G35 alignment).
- Result: wave-parallel dispatch works when each lane's value is
  derived arithmetically from tid + uniform scalars (no vector reads
  needed). This is the building block for RoPE position tables in
  Phase 2 (cos/sin generation is `base + k*i` scaled by a table
  factor), still limited to 8 storing lanes per wave on 9.40.

## G56 wave-parallel ramp, s_load-fed (2026-08-17, run 011947)

Same kernel as G55 but k/base are read from a GPU-mem header via
`s_load_dword` (`s_load_dword s16, s[2:3], 0` / `, 4` + `s_waitcnt
lgkmcnt(0)`), instead of user-data scalars. HW-validated PASS. This
proves the scalar-read data path works inside a 32-thread wave — the
x-side of a wave-parallel GEMV (x[k] is uniform across rows and
readable with scalar loads). ABI: s2:s3 = header (k, base), s4:s5 =
C; commit `adb11b6`.

## G57-G64 bisection: per-lane data select CLOSED (2026-08-17)

Goal: per-lane selection out of a uniform `s_load_dwordx16` block
(the missing piece for wave-parallel GEMV, W-row access per lane).
Final matrix (run 095951, commit fa81eec):

| exp | kernel | RSRC1 | threads | read | result |
|-----|--------|-------|---------|------|--------|
| G58 | blockdump | BLOCK32 0x602C0043 (32VGPR+32SGPR) | 1 | v0..v7 <- s16..s23 | **PASS** c=0x10000000+i |
| G62 | vpick2 | BLOCK32 | 32 | v8 <- s16 | **PASS** c=h[0] x8 |
| G59 | vpick | BLOCK32 | 32 | v16 <- s16 | FAIL c=0..7 (tid) |
| G63 | vpick | BLOCK32 | 1 | v16 <- s16 | FAIL c[0]=0 (tid) |
| G60 | vpick2 | STANDARD 0x602C0000 (8VGPR+16SGPR) | 32 | v8 | FAIL 4i+3 (confounded) |
| G61 | vpick | STANDARD | 32 | v16 | FAIL 4i+3 (confounded) |
| G57 | lanepick | BLOCK32 | 32 | v_movrels m0=64/16 | FAIL tid |
| G64 | movrels | BLOCK32 | 32 | v_movrels m0=7, blk v7..v14 | FAIL c=h[0] uniform |

Conclusions:

1. **Hard VGPR ceiling at 16**: v0..v15 usable, v16+ READS return tid
   and WRITES are dropped, regardless of thread count (G59 vs G63)
   and regardless of the RSRC1 VGPRS field (BLOCK32 declares 32 VGPRs
   but v16+ still reads tid). VGPRS is not a real contract on 9.40.
2. **`v_movrels_b32` is uniform-relative**: G64 moved the block to
   v7..v14 (in-ceiling) with m0=7 and got c=h[0] for ALL lanes — it
   reads `v[regno+m0]`, i.e. a fixed register per instruction, NOT
   `v[m0+tid]`. It cannot do per-lane indexed selection. (G57's
   earlier failures were the v16+ ceiling, but even in-ceiling the
   instruction cannot select per lane.)
3. **G60/G61 were confounded** (B review note 1): STANDARD RSRC1 has
   VGPRS=0 (8 VGPR) AND SGPRS=0 (16 SGPR), so both the v8/v16 reads
   and the s_load_dwordx16 into s[16:31] were out of range. The 4i+3
   leak is the value-path formula, not data.
4. **LDS/DS roundtrip also dead** (G25/G26/G27): ds_write+ds_read
   returns 0 / leaks, both 1KiB and 8KiB LDS_SIZE. The DS fallback
   B proposed is already falsified on 9.40.

**Bottom line**: on 9.40 there is NO per-lane data-select mechanism —
vector loads hang, LDS returns 0, movrels is uniform-relative, VGPRs
stop at 15. Wave-parallel kernels can only derive values
ARITHMETICALLY from tid + uniform scalars (G55/G56 ramp). The
validated serial G40 GEMV (1 thread per group, s_load per element,
groups_x = M) remains the ONLY data-access GEMV path; per-lane W-row
access for a wave-parallel GEMV is impossible on 9.40. RoPE position
tables (G55/G56 ramp) remain the practical wave-parallel payoff.

## G65/G66 wave-parallel cos/sin ramp (2026-08-17, run 101812)

RoPE position-table primitives unlocked: `cossin.s` computes
`c[i] = cos/sin(scale * i)` on the G55 wave-parallel path (32 threads,
lanes 0..7 store) and validates `v_cos_f32` / `v_sin_f32` on 9.40.
HW-validated PASS (commit a8d9089).

- ABI (RSRC2 0x0C): s2:s3 = C, s4 = scale (float); two entries
  (cos @0, sin @16, 16 words each) in one `pai_cossin.inc`.
- **NEW RULE - turns convention**: `v_cos_f32`/`v_sin_f32` read their
  operand in FULL TURNS, not radians — the VALU value path multiplies
  by 2*pi. HW evidence: theta = 0.15*(4i+3) in radians gives
  cos(162°)=-0.9511 exactly (i.e. the silicon computed
  cos(2*pi*0.15*3)); all 16 values (8 cos + 8 sin) matched
  `cos/sin(2*pi*scale*(4i+3))` to <1e-4 with cos^2+sin^2=1.
  The oracle/mirror must use the x2pi convention.
- Combined with the G15 value-path quirk (lane index reads (4i+3)),
  the effective GPU formula is `c[i] = cos/sin(2*pi*scale*(4i+3))`.
- Implication for Phase 2: cos/sin tables for RoPE can be generated
  on-GPU (wave-parallel) with scale = inv_freq/(2*pi) per column;
  matches B's `pai_ref_rope_cossin_f32` oracle (theta = p*inv_freq)
  once the turns convention and (4i+3) indexing are mapped.

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

## G67-G70: on-GPU RoPE table generator (serial-per-element)

- G67 (cos, commit d502b10): HW-VALIDATED — groups_x = ctx*r2 (128),
  1 thread/group, TGID_X = e = p*r2+i, theta_turns[e] host-computed
  (= theta/2pi so the turns convention cancels), s_load per element
  (G40 pattern), v_cos/v_sin, flat store at C[0..ctx*r2). All 128
  values exact vs pai_ref_rope_cossin_f32 oracle.
- G68 (sin): FAIL — but NOT a GPU-state flake. EOP label NEVER fires
  (30s x 3 fences), first submission zero writes in 30s, re-submit
  ~112/128 exact writes, holes at e=13..15, 22..29, 32..36 + occasional
  extras; queue recovers afterwards (G17+26 run). M0-B also fails in
  every post-panic run, which initially suggested a wedged GPU.
- G69 (sin kernel, v_sin word 17 patched 7E026B01 -> 7E026D01 = v_cos):
  PASS, bad=0, all 128 exact — in the SAME run where G68 fails.
  Conclusion: SERIAL-PER-ELEMENT v_sin_f32 IS TOXIC on 9.40 (waves
  hang, dispatch never retires, EOP never fires) while v_cos_f32 is
  fine. G65/G66 only validated v_sin wave-parallel (1 group, many
  lanes) — the serial form was never exercised before.
- G70 (sin via cos-shift): PASS, bad=0 — dispatch the cos path with
  theta_turns' = theta_turns - 0.25 (i.e. cos(2pi*(tt-0.25)) =
  cos(theta-pi/2) = sin(theta)), writing the sin half. Production
  sin-table path validated.
- Empirical rule 9.40: v_cos_f32 OK serial; v_sin_f32 OK wave-parallel
  (G66) but TOXIC serial-per-element. RoPE tables on GPU: cos = G67
  direct, sin = G70 cos-shift.

- G71/G72 (run 132918): serial-per-element v_rsq_f32 / v_exp_f32.
  v_rsq: PASS, 64/64 exact 1/sqrt(x) (0.0625..4.0) - RMSNorm's
  inverse-root path is clean serial-per-element. v_exp: PASS as
  2^x (bad_2=0, bad_e=63, x in -4..3.875) - GCN convention
  CONFIRMED on 9.40; softmax/SiLU e^x must pre-scale x by
  log2(e) ~= 1.442695 on the validated mul path (same shape as the
  cos/sin turns workaround).

- Working: bootstrap, jailbreak, /data logging, lifecycle listener,
  notify, DMA, fence, dispatch, SMEM loads, integer ALU, stores,
  add1d / SAXPY / serial dot / serial GEMV on the G22 path; float ALU
  (G35/G39/G40/G41); T4 serial kernel family G42-G54 float + integer
  (13/13, run 004416); G65/G66 wave-parallel cos/sin ramp (turns
  convention); G67 serial on-GPU RoPE cos table; G70 cos-shift sin
  table;  G71 v_rsq; G72 v_exp (2^x); G73 decoder GEMM via G40 (per-row,
  transposed-W repack hdr[4+g*K+k]=B[k*N+g])  8x16x8 PASS 64/64 vs
  double oracle - same ABI decoder_test sez.12 drives on host.
  G74 (run 135700): FIRST on-GPU decoder prefill forward - QKV/out/
  MLP/logits via G40 per-row + RoPE cos/sin tables via G67/G70 path,
  attention/RMSNorm/SiLU/residual host ref_ops; logits == host ref
  chain (  1e-4, mism 0), argmax 2 == 2 (PASS). The decoder_test
  sez.12/13 differential contract now HW-proven on 9.40.
  G75/G76 (run 141157): v_rcp_f32 serial PASS 64/64 exact 1/x
  (0.5..16.25 - softmax/SiLU denominator range); v_max_f32/v_min_f32
  serial PASS bad=0. Closes spec sez.10 risks (v_rcp/v_max) - all
  softmax/SiLU primitives now HW-proven: division via v_rcp,
  max pass via v_max, exp via 2^x prescale, rsqrt via v_rsq.
  G77 (run 144815): FULL autoregressive decode loop on-GPU - prefill
  + 4 generated tokens via G40 GEMMs (per-row, transposed-W) +
  G67/G70 RoPE tables (loaded once, reused at every step, p grows),
  attention/RMSNorm/SiLU/residual host ref_ops, oracle
  m0_decoder_ref_chain (full-prefix re-forward with same weights);
  every step tok==ref_argmax, logits mism=0, bad_steps=0 (PASS).
  First-token -> next-token loop closed on 9.40 (decoder_test sez.13
  contract). Convention locked: residual adds the RoPE-ROTATED
  embedding (oracle applies rope_f32 in-place), not the raw embed.
  G78 (run 145847): serial causal attention ON-GPU (spec §6 / B's
  attention_serial contract 32a3eca) - per (row, head): scores via
  G40 gemv (groups=p+1, W transposed hdr[4+t*hd+d]=k[t][d]), scale
  1/sqrt(hd) + row-max reduce host, e=2^((s-m)*log2e) via REAL G72
  nlexp_exp dispatch (prescale host), sum host + rcp via REAL G75
  nlexp_rcp dispatch, soft=e*rcp, out via G40 (W row-major
  hdr[4+d*(p+1)+t]=v[t][d]). PASS vs pai_ref_attention_f32 (H2 HK1
  HD4 seq6, 1e-4, mism=0). The LAST host stage of the decoder
  forward (attention) is now on-GPU on 9.40.
  G79 (run 150645): RMSNorm ON-GPU (spec §4) - per row: s = sum x^2
  via G40 dot (W = x, groups = 1), inv = 1/sqrt(s/n+eps) via REAL
  G71 nlexp_rsq, c = x*(gamma*inv) via G42 t4_mul1d. PASS vs
  pai_ref_rmsnorm_gamma_f32 (4 rows x 8, 1e-4, mism=0).
  G80 (run 150645): SiLU ON-GPU (spec §5) - prescale xs = -x*log2e
  host, e = 2^xs via REAL G72 nlexp_exp, d = 1+e via G42 t4_add1d,
  rc = 1/d via REAL G75 nlexp_rcp, c = x*rc via G42 t4_mul1d. PASS
  vs pai_ref_silu_f32 (16 elems, 1e-4, mism=0). All 3 formerly-host
  decoder stages (attention G78, RMSNorm G79, SiLU G80) now on-GPU:
  decoder forward is 100% GPU on 9.40.
- Open (do not block M1 closeout): MUBUF T# / flat loads, lane 8+
  exec mask, LDS (M1C) pending AGC CS blob @ 0x213.


