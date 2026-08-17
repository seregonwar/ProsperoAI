# Phase 2 — GPU decoder forward: dispatch sequencing spec

**Status:** spec for Seat A's decoder-path integration (whitepaper §43)
**Author:** Seat B (Buffy)
**Date:** 2026-08-17
**Oracle:** `toolchain/decoder_test.c` sections 12 (prefill) and 13
(autoregressive loop) — both PASS vs the double oracle at 1e-4 + argmax.
**Reference topology** (decoder_test): `D_MODEL=8`, `H_HEADS=2`,
`HK_KV=1`, `HD_DIM=4`, `R2=2`, `SEQ=3`, `MLP_HID=16`, `VOCAB=16`.
All formulas below are dimension-parameterized; the values in
parentheses are the reference-topology numbers.

---

## 1. Objective

Produce the first forward pass of the small decoder with the linear
layers and the transcendental ops running on the real 9.40 GPU, and
differentially validate it against the host oracle. Success criteria:

- prefill logits match `decoder_test` sez.12 oracle at `1e-4`
  (relative, `pai_ref_compare_f32` semantics) with identical greedy
  argmax;
- the same dispatch plan extended to the decode loop matches sez.13.

## 2. Kernel inventory (all HW-validated on 9.40)

| Kernel | File | ABI (ud[2:3] header, ud[4:5] out) | Validated |
|---|---|---|---|
| G40 float GEMV serial-per-row | `fgemv_serial.s` | `[K, pad, x_lo, x_hi, W...inline at hdr+4]`; `groups_x = N` rows; `y[g] = Σ_k Wt[g*K+k]·x[k]` | G40, G73 (64/64) |
| G48 float matmul per-cell | `t4_ops.s` | `[K, N, a_lo, a_hi, b_lo, b_hi]`; `groups_x = M*N` | G48 |
| G47 biasadd | `t4_ops.s` | `[cols, pad, a_lo, a_hi, bias_lo, bias_hi]`; `groups_x = rows*cols` | G47 |
| G42 add1d (residual) | `t4_ops.s` | packed `(a,b)` at ud[2:3], C at ud[4:5]; `groups_x = n` | G42 |
| G67 ropegen cos | `ropegen.s` | `[r2, ctx, theta_turns[0..ctx*r2-1]]`; `groups_x = ctx*r2`; writes cos half | G67 (128/128) |
| G70 ropegen sin = cos-shift | `ropegen.s` | same header, **theta_turns − 0.25**; `v_cos` instruction; writes sin half at `+ctx*r2` | G70 |
| G71 v_rsq | `nlexp.s` | `[n, pad, x[0..n-1]]`; `groups_x = n` | G71 |
| G72 v_exp (= 2^x) | `nlexp.s` | same ABI | G72 |

### 9.40 empirical rules that shape every dispatch
- **No per-lane data select.** Only uniform `s_load` + arithmetic
  values derived from tid. `v_movrels` is uniform-relative, LDS/DS
  dead, VGPR ceiling 16 (never touch v16+).
- `v_sin_f32` is **toxic serial-per-element** → sin only via cos-shift.
- `v_exp_f32` computes **2^x** → e^x = 2^(x·log2(e)), pre-scale on host.
- `v_cos_f32`/`v_sin_f32` take **turns** (×2π in the value path) →
  feed `theta/(2π)` = `theta_turns`.
- Lane quirk G15: wave-parallel value paths see `(4·tid+3)`, not `tid`
  — serial kernels (1 thread/group) are immune.

## 3. Op → dispatch mapping (prefill)

Linear layers use **G40**, one dispatch per input row
(`groups_x = N`), with W repacked **transposed inline**
(`hdr[4+g*K+k] = B[k*N+g]`), per-row C base `c + i*N` (floats). Host
does the repack once per layer (weights are static across tokens).

| # | Op | Kernel | Dispatches | groups_x | Notes |
|---|---|---|---|---|---|
| 1 | RoPE cos table | G67 | 1 | `ctx*r2` | header theta_turns (host-computed) |
| 2 | RoPE sin table | G70 | 1 | `ctx*r2` | header theta_turns − 0.25 |
| 3 | Q = E·Wq | G40 | `SEQ` | `D_MODEL` | x = RoPE'd E row i |
| 4 | K = E·Wk | G40 | `SEQ` | `D_MODEL` | |
| 5 | V = E·Wv | G40 | `SEQ` | `D_MODEL` | |
| 6 | attention (causal GQA) | **host** | — | — | no validated kernel (softmax serial plan below) |
| 7 | O = Attn·Wo | G40 | `SEQ` | `D_MODEL` | |
| 8 | residual + (host add or G42) | G42 | 1 | `SEQ*D_MODEL` | packed (a,b) |
| 9 | RMSNorm | G71-based | `SEQ` | see §4 | |
| 10 | gate = h1·W1 + b1 | G40 + G47 | `SEQ` + 1 | `MLP_HID` / `SEQ*MLP_HID` | |
| 11 | SiLU | G72-based | 1 | `SEQ*MLP_HID` | pre-scale −x·log2e on host |
| 12 | mlp = silu·W2 + b2 | G40 + G47 | `SEQ` + 1 | `D_MODEL` / `SEQ*D_MODEL` | |
| 13 | residual + | G42 | 1 | `SEQ*D_MODEL` | |
| 14 | RMSNorm | G71-based | `SEQ` | §4 | |
| 15 | logits = h·Wout | G40 | `SEQ` | `VOCAB` | |

Total G40 dispatches (prefill, reference): 5 projections × 3 rows +
MLP W1/W2 × 3 + logits × 3 = 21; plus 2×G42, 2×G47, 2×G67/G70,
SEQ×RMSNorm(2 ops each), 1×G72. All kernels serial — no wave-parallel
dependency anywhere except the tables (G67/G70 use the 32-thread wave
with the `(4l+3)` mapping; rows written are p = 3,7,…,31 — compare
**only** written rows).

## 4. RMSNorm serial recipe (G71)

For each row of `n` elements (n = D_MODEL or MLP_HID):
1. **mean-square**: one serial dot pass — G40-style kernel or reuse
   the G48 dot: `s = Σ x[i]·x[i]` (float accumulation; margin vs
   double oracle ≤ 5.3e-06 worst case, budget 1e-4 safe).
2. **inverse root**: `G71` on the scalar `s/n + eps` → `inv`.
3. **scale**: `c[i] = x[i]·γ[i]·inv` — G42 mul (elementwise) or a
   G40 row with W = γ·inv pre-scaled host-side (2 dispatches/row
   worst case: one mul pass).
Float-only margin already contracted in `nonlinear_contract` (§1).

## 5. SiLU serial recipe (G72, 2^x)

`σ(x) = x / (1 + 2^(−x·log2e))`:
1. host pre-scale `xs = −x·log2(e)` into the G72 input buffer (float
   mul, the G39-validated path);
2. G72 → `e = 2^xs`;
3. `1/(1+e)` + `·x` — one elementwise pass (G42 mul/add chain or
   G40 rows). Contract: `exp2f(x·log2e) == expf(x)` max rel
   1.18e-06, softmax row sum 1.000000041 in 2^x form.

## 6. Softmax (attention rows) — serial plan (future)

No validated kernel yet; the serial recipe from `nonlinear_contract`
§3/§5: max pass → `2^((s−m)·log2e)` via G72 → float sum → reciprocal
+ mul. Cost per row `O(seq)` dispatches. **For the first GPU forward,
keep attention on host** (sez.12 does exactly this); the softmax GPU
leg is a follow-up once `v_rcp`/division is probed.

## 7. Buffer layout

- One GPU-resident activation arena, position-major `[seq][D_MODEL]`
  per tensor (E, Q, K, V, Attn, O, h1a, h1, gmlp, ag, h2, mlp, h3a,
  h3, logits).
- Per-GEMM W repack buffers: `4 + N*K` u32 words each, built host-side
  once per weight (Wq/Wk/Wv/Wo/W1/W2/Wout). **Header buffer type
  must be u32** (pointer halves); W payload floats via `memcpy`
  (bit-preserving) — do not store u32 pointer halves into a float
  array (ASan-caught bug, commit `cc8f261`).
- RoPE: `theta_turns[ctx*r2]` and `theta_turns−0.25` header arrays;
  cos/sin tables `[2][ctx*r2]` (sin at +ctx*r2).
- G71/G72 input/output buffers (reusable scratch).

## 8. Dispatch sequence invariants

1. Every dispatch sets **both** header (ud[2:3]) and output
   (ud[4:5]) pointers — stale ud[4:5] corrupts shared buffers
   ("opcode 0xff" bug, commit `1cfceb5`).
2. One thread per group for serial kernels (`NUM_THREAD_X=1`,
   `RSRC2 0x8C` pattern); only G67/G70 use the 32-thread wave.
3. Fence/EOP per dispatch or batch; G67-style wait before reading
   tables; the GPU is known to wedge after panic — validate tables
   on a healthy console state.
4. Order: tables (1,2) → QKV (3–5) → attention host → out (7) →
   residual (8) → RMSNorm (9) → MLP (10–12) → residual (13) →
   RMSNorm (14) → logits (15).

## 9. Differential contract

- Prefill: logits == `decoder_test` sez.12 oracle, `1e-4` rel + greedy
  argmax equal.
- Decode: per-step logits == sez.13 full-prefix oracle (KV cache
  storage already covered by the `pai_kv_cache` manager section 10).
- Tables: compare **only** written rows (p = 4l+3), cos²+sin²=1,
  `base=10000` explicit.

## 10. Risks / open probes

- ~~`v_rcp`/division untested~~ — **CLOSED (G75, commit `78a8595`):**
  `v_rcp_f32` serial-per-element PASS 64/64 exact (1/x on
  0.5..16.25, the softmax/SiLU denominator range). Softmax division
  and `1/(1+e)` are directly buildable — no G71 workaround. Locked
  host-side in `nonlinear_contract` sez.6: softmax e·rcp(sum)
  (max |d| 5.96e-08, row sum 0.999999954), SiLU x·rcp(1+e)
  (2.38e-07), both through the exact nlexp mirror ABI.
- ~~`v_max_f32` untested~~ — **CLOSED (G76, commit `78a8595`):**
  `v_max/v_min_f32` PASS bad=0 (softmax max pass ok);
  `v_max(x,0) == relu` locked host-side (exact). Note: v_max is
  ELEMENTWISE (header [n,pad,x,y]); the softmax row-max is a serial
  reduce pass over it, same shape as the dot-sum.
- Per-row G40 means 21+ dispatches per prefill — dispatch overhead is
  the known serial-model cost (apparent ~2–4 GB/s signal only).
- Attention on host for the first GPU forward is the agreed scope;
  full on-GPU attention is a follow-up with the §6 serial plan — all
  its primitives (max/rcp/exp-2^x/mul-add) are now HW-validated.
