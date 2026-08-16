# ProsperoAI — Phase 1: Tensor Runtime Design

**Status:** Design (v0.1)
**Depends on:** Phase 0 (PAI-M0 / PAI-M1 hardware bring-up) — CLOSED on the serial G22 path
**Scope:** the complete Phase 1 tensor runtime, from tensor descriptors to the first end-to-end synthetic network
**Related:** `ProsperoAI — Technical Whitepaper v0.1.md` (§3, §10–13, §16–18, §37), `notes/re/940-gpu-empirics.md`

---

## 1. Purpose

This document is the complete design of the Phase 1 deliverable from the
whitepaper roadmap (§40):

> Tensor descriptors; datatypes; memory planner foundation; reference CPU
> backend; GPU backend; first optimized operators; profiling.
> **Target: run a small synthetic neural network completely through
> ProsperoAI.**

Phase 1 does not aim for LLM-grade performance. It aims for a **correct,
end-to-end, profiled execution path** that exercises every layer of the
runtime on real PS5 hardware, and that can absorb the GPU unlocks
(MUBUF/vector reads, float ALU, LDS) without redesign when they land.

Everything here is constrained by the empirical silicon rules in
`notes/re/940-gpu-empirics.md`. Section 3 summarizes the constraints that
shape the design; do not design around them, design **within** them.

---

## 2. Scope Boundaries

In scope:

- dense tensor descriptors (row-major) and dtype tables;
- static memory planner foundation (arena + lifetime-based placement);
- CPU reference backend as the correctness oracle;
- GPU backend built on the proven serial scalar dispatch path;
- a minimal operator set sufficient for a synthetic network;
- profiling (wall-clock, per-op, per-device);
- one end-to-end synthetic model (MLP-style) on both backends.

Out of scope for Phase 1 (later phases):

- quantization kernels and sub-byte formats (metadata hooks only);
- KV cache, tokenizer, sampling, streaming;
- autotuning, kernel IR codegen, operator fusion beyond trivial
  elementwise fusion;
- the Prospero Protocol / gateway / desktop pipeline (they exist, but
  Phase 1 does not depend on them);
- the MUBUF/flat/float/LDS GPU unlocks (tracked, not required — see §11).

---

## 3. Hardware Reality Baseline (9.40)

The Phase 1 GPU backend must be designed around what the silicon
**provably** does today. Every claim below was verified on hardware
(see the empirics notes for the experiment IDs).

Working compute path (the "serial G22 path"):

- submission: `/dev/gc` SUBMIT_16, queue type 3, const-IB CB descriptor
  + 16-dword NOP trailer, action-based IT_RELEASE_MEM EOP fence;
- dispatch: SET_SH_REG (compute bank) + DISPATCH_DIRECT, RSRC1
  `0x602C0000`, 4–9 user SGPRs, W32 waves;
- **scalar memory reads**: `s_load_dword` / `s_load_dwordx16` from the
  GPU VA (SMEM path) — fully functional, CPU-reference-checked (G22,
  G24, M1A/M1B/M1D kernels);
- **stores**: `flat_store_dword` with vaddr pair `v[2:3]`; data = the
  last SGPR-sourced value + `tid*4 + 3`, or a literal written to v0
  verbatim; only lanes 0–7 of a 32-thread wave store (exec-mask
  quirk);
- **per-thread integer ALU** with `dst != v0` (lshl, add_co), result
  copied to v0 for the store;
- buffer fills: WB_ONION dmem (type 1) + CPU-side writes are
  coherent; WC_GARLIC (type 3) requires clflush before GPU reads;
- fencing: label poll on dmem after the EOP action.

Broken / open on 9.40 (do not build on them):

- **flat_load_dword: HANGS the wave and the ring.** Never emit it.
- **buffer_load_dword (MUBUF): completes but zero-fills** for every
  T# word3 candidate tried so far. The vector read path is not usable
  yet; the T# hunt continues in parallel (§11).
- **float ALU: UNLOCKED (2026-08-16).** The VALU float form is
  _cvt_f32_i32 (int->float) + _add_f32_e64/_mul_f32_e64 with
  direct SGPR operands + VGPR+VGPR accumulator, dst != v0. Validated:
  float add (G35), float dot (G39, exact vs CPU), float GEMV with
  parallel rows (G40), float SAXPY per-group (G41). SALU float does not
  exist on gfx1013 (float is VALU-only). Avoid: dst-v0 float ops (hang),
  e32 VOP2 mixed VGPR+SGPR (VGPR zeroed), _mov_b32 vX, s4 (SGPR read 0).
- **LDS (ds ops): writes/reads complete alone, but any s_load + ds
  combination hangs** (G26/G27). The parallel-reduction path is
  blocked pending a real AGC CS RSRC2 dump. Do not invent further
  LDS_SIZE values.
- **User-range SGPR writes (readfirstlane into s2–s9) hang.** Scalar
  targets must be beyond the user SGPR window.
- **s_barrier with 8-of-32 active lanes deadlocks.** Avoid barriers.
- apparent GPU↔GPU bandwidth ≈ 2–4 GB/s (GEMV measurement), not HBM
  speeds. Budget accordingly; do not promise HBM throughput.

Design consequences:

1. Every GPU operator is a **serial scalar kernel**: the dispatcher
   walks the output in fixed-size chunks (16 elements per
   `s_load_dwordx16`), the wave's lanes 0–7 compute 8 of them per
   dispatch, and the results are written back with `flat_store_dword`.
2. The GPU backend is **correctness-first**: the CPU reference backend
   is the ground truth; the GPU path must match it bit-exactly for
   integer ops and within tolerance for the future float ops.
3. All arithmetic is **i32/u32 on the GPU** until the float ALU is
   unlocked; the synthetic network uses integer/quantized-equivalent
   datatypes on the GPU and f32 on the CPU reference.
4. The operator interfaces must not bake in the serial model: the
   kernel selection layer (§7) exists so the future vector/MUBUF/LDS
   kernels can be substituted per-operator without touching the graph
   or the planner.

---

## 4. Layered Architecture

```text
Application (harness / synthetic NN)
        |
   Runtime API (pai_runtime_*)
        |
   +----- Execution plan (static) -----+
   |  graph nodes -> op registry      |
   |  -> device placement             |
   |  -> memory planner               |
   +----------------------------------+
        |                        |
   CPU Reference backend    GPU backend (serial G22)
        |                        |
   tensor + dtype + arena    hal (buffer alloc / submit)
                             kernels (llvm-mc gfx1013)
                             profiling (timers, labels)
```

Data flow rule: **tensors are views over storage; storage is owned by
the runtime arena.** A backend never allocates tensor memory itself;
it receives buffer bindings from the planner.

---

## 5. Tensor Descriptors and Datatypes

The existing `tensor/` module (`pai_tensor_t`: dense row-major,
rank ≤ `PAI_TENSOR_MAX_RANK`, strides computed from the shape, dtype
via the shared dtype table) is the foundation. Phase 1 extends it
without breaking it.

### 5.1 dtype table

Storage dtypes (already present): f32, f16, bf16, i32, u32, i16, u16,
i8, u8.

Phase 1 additions:

- `PAI_DTYPE_Q4_0`-style **quantized metadata descriptors** — not
  storage primitives, but a `pai_quant_t` layer: block size, scale
  dtype, zero point, packing order. Quantization is metadata-only in
  Phase 1 (no quantized kernels yet); the field exists so the `.pai`
  container and the future kernels share one vocabulary.
- an **accumulator dtype** field for ops that upcast internally
  (i32 for i8/u8 matmuls, i64 for i32 reductions) — avoids silent
  overflow in the serial dot products.

### 5.2 descriptor invariants

- `data == NULL` is a valid **virtual tensor** (shape-only); the
  planner materializes it lazily. This is how the graph compiler
  expresses intermediate activations without pre-allocating.
- strides are always dense row-major in Phase 1; a `pai_tensor_view`
  function (offset + size clamp) supports the serial kernel chunks
  without copies.
- every tensor records `device` (CPU / GPU) and `storage_id` (arena
  slot); backends use these to decide whether a transfer is needed.

### 5.3 layout normalization

- All tensors are normalized to **contiguous row-major** before
  kernel dispatch. Transpose, reshape-as-view and strided reads are
  CPU-side helper ops (materializing) for Phase 1; kernel-side
  layout specialization is a later-phase optimization.

---

## 6. Memory Planner Foundation

The planner answers: *which buffers exist at the same time, and where
do they live?* Phase 1 implements the static subset only.

### 6.1 model

- Input: the execution plan's op list with per-tensor sizes, dtype,
  device and liveness intervals (producer-to-last-consumer).
- Output: `pai_plan_allocation_t[]` — arena offset + size + device
  per tensor, or an error when the budget is exceeded.

### 6.2 algorithm

Phase 1 uses **first-fit with lifetime reuse** over a single arena per
device (CPU arena = malloc-backed; GPU arena = dmem-backed, 2 MB
granularity, WB_ONION):

1. sort buffers by (start, -size);
2. place into the lowest free span that fits (liveness-aware);
3. honor 2 MB alignment on the GPU arena (dmem granularity);
4. pin model weights to persistent slots (never reused).

Weights go to persistent slots; activations to transient slots that
may alias across layers. The planner output is deterministic and
recorded in the profile log (§9) so replays are reproducible.

### 6.3 GPU budget modes (foreshadowing §17 of the whitepaper)

The planner takes a single `pai_mem_mode_t` in Phase 1:

- `PERFORMANCE` — everything GPU-resident (needs the full footprint);
- `BALANCED` — weights GPU-resident, activations may spill to CPU.

Phase 1 implements PERFORMANCE only; BALANCED requires the transfer
engine (Phase 7 scope), but the enum and the planner interface are
designed for it now.

---

## 7. Backends

### 7.1 CPU Reference Backend (`cpu/reference`)

The correctness oracle (whitepaper §37). Properties:

- straight C, no intrinsics, deterministic;
- every operator has an exact reference implementation;
- integer ops must match the GPU bit-exactly; float ops are compared
  with configured tolerance;
- differential testing: the same plan runs on CPU and GPU and the
  tensors are compared element-wise;
- doubles as the fallback for any unsupported op in the eager mode.

### 7.2 GPU Backend (`gpu/`)

The **serial scalar dispatch model** (G22-proven), generalized:

```text
op(ctx, out_chunk):
  for chunk in 0..ceil(N / 16):
      ud = { in_base[lo,hi] (+chunk offsets), out_base, params }
      dispatch(16 elements via s_load_dwordx16)
      lanes 0-7 compute 8 elements (integer ALU, dst != v0)
      flat_store_dword writeback (v[2:3], literal-v0 pattern)
      label poll (EOP action fence)
```

Kernel inventory (already validated on hardware in PAI-M1):

| kernel | signature | status |
| --- | --- | --- |
| `add1d` | c[i] = a[i] + k | VALIDATED (N = 1..1M) |
| `saxpy_u32` | c[i] = a[i] + k·b[i] (integer) | VALIDATED |
| `dot_serial_u32` | reduction via serial accumulation | VALIDATED (correctness) |
| `gemv_serial_u32` | per-row serial dot | VALIDATED (256×1024) |

Phase 1 adds, on the same serial pattern:

- `mul1d`, `sub1d`, `relu_u32` (max with 0), `clip_u32`;
- `add2d` (elementwise sum of two vectors);
- `matmul_u32` = generalized `gemv_serial_u32` (blocked: rows ×
  k-chunks, i32 accumulator);
- `softmax_u32` (serial max → exp-free integer approximation for the
  exit test, or CPU-side softmax — see §10).

Submission plumbing reused verbatim from PAI-M1: `pai_gc_submit`,
the EOP action fence, the label poll, the `pai_gpu_reset` between
experiments. One kernel source file per op family under
`gpu/kernels/gfx1013/`, assembled with llvm-mc 18 via
`toolchain/assemble_shader.py`, and mirrored by a host reference
implementation for the host-ref interpreter (keeps `make host` green).

### 7.3 kernel selection

`pai_op_exec` never calls a kernel directly:

```c
pai_kernel_vtable_t *pai_kernel_select(const pai_op_t *op,
                                       const pai_device_t *dev);
```

Phase 1 registers exactly one GPU implementation per op (the serial
kernel). The vtable indirection is the seam where the future
vector/MUBUF/LDS kernels will be inserted when §11 unlocks land —
no graph or planner changes required.

---

## 8. Operator Set and the Execution Model

### 8.1 op registry

A table-driven registry: `{ name, version, in/out signature, device
caps, reference fn, gpu kernel id }`. Ops are looked up by name +
signature hash. Unknown ops in the eager mode fall back to the CPU
reference automatically.

### 8.2 execution model

Phase 1 runs **static plans**: the graph is lowered once into
`pai_plan_t` (topological op list + memory assignments), then executed
repeatedly. An eager convenience path (`pai_op_exec` on single ops)
exists for the harness and the tests.

### 8.3 the Phase 1 operator set

Elementwise: add, sub, mul, relu, clip, copy.
Reduction: dot (serial), l1/l2 norm (for the profile logs).
Linear algebra: gemv, matmul (serial/blocked), bias-add.
Shape: reshape (view), concat (copy).
Normalization: softmax (see §10).

This is deliberately small: enough for an MLP, small enough to
reference-check exhaustively.

---

## 9. Profiling

Phase 1 profiling answers one question per op: **where does the time
go?** Three timers:

1. CPU wall-clock per op (CLOCK_MONOTONIC, ns);
2. GPU wall-clock per op (label poll latency, submit-to-fence);
3. transfer counter (CPU↔GPU copies, bytes + count).

Output: a per-plan profile log (JSON-lite text) with per-op
(durations, tensor sizes, device, kernel id) and the planner's memory
summary. Deterministic replay: the profile records the seed and the
plan hash.

The 2–4 GB/s GPU bandwidth baseline is the number the Phase 2
estimates will extrapolate from, so the Phase 1 profiler must be
honest about it (no wall-clock inflation from the poll loop — the
timer starts at submit and ends at the fence).

---

## 10. The Exit Test: Synthetic MLP

The Phase 1 acceptance test is a small MLP: `8 → 16 → 8` (i32 weights,
relu hidden layer, softmax output), fixed seed, run on:

1. the CPU reference backend — exact;
2. the GPU backend (serial kernels) — integer ops bit-exact vs the
   CPU; softmax compared within tolerance (or computed CPU-side with
   a GPU↔CPU transfer, whichever the current ALU state allows);
3. the host-ref interpreter (desktop CI) — same plan, same numbers.

Pass criteria:

- all three runs agree within the configured tolerance;
- the plan is fully memory-planned (no runtime malloc in the hot
  path);
- the profile log is produced and plausible (GPU ops ≥ 1, transfers
  accounted, no negative durations);
- the whole test runs inside the deploy loop on hardware (the
  PAI-M1 harness pattern).

This test becomes the regression gate for every later kernel change.

---

## 11. Blocked Work and Unlock Paths

Tracked, not required by the Phase 1 exit criteria:

| item | blocker | unlock |
| --- | --- | --- |
| MUBUF / vector reads | T# zero-fills for all candidates | capture a real AGC shader CS blob; diff T# words; retest per-thread loads |
| flat loads | ring hang | same as above (likely the same VM read-side issue) |
| float ALU | ~~returns 0 with dst != v0~~ | **DONE** - e64 direct-SGPR form (G35/G39/G40/G41) |
| LDS staging / M1C parallel reduction | `s_load` + `ds` combo hangs | real CS RSRC2 dump (SH offset 0x213); then barrier + LDS re-test |
| bandwidth | ≈ 2–4 GB/s apparent | profile after the vector-read unlock; investigate async/pipelined submits |

Rule: do not block Phase 1 on any of these. The serial kernels are
the correctness path; the unlocks replace them per-op through the
kernel vtable.

---

## 12. Work Breakdown

1. **T1 — dtype/quant metadata** (tensor/dtype.c): quant descriptor
   struct, accumulator dtype field, tests.
2. **T2 — planner v1** (memory/planner.c): lifetime first-fit, GPU
   arena alignment, weight pinning, budget check, tests.
3. **T3 — CPU reference ops** (cpu/reference): the §8.3 op set with
   deterministic implementations + tolerance helpers.
4. **T4 — serial GPU kernels** (gpu/kernels/gfx1013): mul1d, sub1d,
   relu_u32, clip_u32, add2d, matmul_u32; each with a host-ref
   mirror; hardware-validated through the deploy loop.
5. **T5 — op registry + static plan executor** (graph/, scheduler/):
   the topological plan, the kernel vtable, the eager fallback.
6. **T6 — profiler** (profiler/): the three timers, the plan profile
   log, the replay hash.
7. **T7 — synthetic MLP harness** (payload or a new binary): the exit
   test of §10, run on hardware + host + CI.
8. **T8 — docs**: update the whitepaper §40 (Phase 1 checkboxes), the
   empirics notes with every new kernel rule discovered during T4.

Suggested order: T1, T2, T3 (host-green, no hardware) → T4 (hardware
loop) → T5, T6 → T7 (exit test) → T8.

---

## 13. Exit Criteria

Phase 1 is complete when:

- [ ] the synthetic MLP runs end-to-end on the PS5 GPU backend and
      agrees with the CPU reference (tolerance per §10);
- [ ] the same plan runs on the host-ref interpreter with identical
      results (CI green);
- [ ] the planner produces a valid allocation for the MLP without
      runtime allocation in the hot path;
- [ ] the profile log is emitted and machine-readable;
- [ ] every new GPU kernel rule discovered during T4 is recorded in
      `notes/re/940-gpu-empirics.md`;
- [ ] the whitepaper roadmap §40 Phase 1 checklist is updated.

Non-goals (explicitly deferred): quantization kernels, KV cache,
autotuning, kernel IR codegen, the MUBUF/float/LDS unlocks.

