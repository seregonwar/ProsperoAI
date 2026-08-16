# ProsperoAI

High-Performance General-Purpose AI Runtime for jailbroken PlayStation 5.

See `ProsperoAI — Technical Whitepaper v0.1.md` for the architecture, and
`notes/re/940-gc-ioctl.md` for the GPU bring-up research.

## Status

**Phase 0 — Hardware Bring-Up (PAI-M0)** in progress.

Implemented so far:

- Monorepo layout per whitepaper §39
- CMake + Ninja build with the §36 presets (`ps5-debug`, `ps5-release`,
  `ps5-safe`, `host-reference`, `host-tests`, `host-sanitized`)
- Tensor descriptors, dtypes, arena allocator, structured logging
- **Memory subsystem foundation** (whitepaper §16): static memory
  planner (lifetime-based buffer reuse, best-fit, alignment-aware) and
  a buddy suballocator (power-of-two split/merge, strict alignment,
  double-free detection) as the GPU region suballocator foundation
- **Graph layer** (whitepaper §10/§11): generic compute graph — values,
  ops, Kahn topological sort with cycle detection, tensor lifetime
  analysis and an integrated static memory plan (§11 memory-planning
  step: non-overlapping lifetimes share storage)
- **Prospero IR** (whitepaper §10.1): standalone graph-level IR — value
  categories (input/param/constant/activation/output), quantization
  metadata (§15), device placement constraints, and a flat binary
  serialization (magic + version + CRC-32, fixed-size records) that is
  the portable form for `.pai` containers and the protocol. Includes a
  faithful graph ↔ IR bridge and a **Kernel IR** skeleton (§10.2:
  workgroup/tile/vector-width/layout descriptors + naive per-op
  lowering; fusion is a later pass).
- **Scheduler / Execution Planner** (whitepaper §18 + §11): turns a
  graph + memory plan into an Execution Plan — topo-ordered steps with
  CPU/GPU placement (per-op-kind defaults, explicit hints, Exclusive
  policy forcing compute onto the GPU), per-step live/produced/released
  byte accounting, §17 memory modes (Performance/Balanced/Capacity) and
  a session registry with priority picking (continuous-batching seed).
  Ships a **reference executor** that runs the plan over the
  memory-plan region on the reference backend — the Phase 1 target
  ("run a small synthetic neural network completely through
  ProsperoAI") is exercised end-to-end by `test_scheduler`.
- **`.pai` Model Container** (whitepaper §20): the native model
  package — magic/version/flags, section table, whole-file CRC-32
  integrity. Portable sections: structured metadata, tensor manifest
  (name → value id, dtype, shape, weight offset), canonical weights,
  embedded Prospero IR, serialized tokenizer. Target slices and
  execution profiles are reserved (§21 fat-binary layout). Reader
  validates magic/version/table bounds/CRC before trusting anything.
- **Model manager + CPU inference** (whitepaper §3.2, §20, §22):
  loads `.pai` containers into executable models (rebuilds the graph,
  computes a **persistent** static memory plan so session weights stay
  resident across repeated execution), then runs sessions on the
  reference executor. Ships the CPU-side **tokenizer** (byte-level BPE:
  vocab + merges + byte fallback, serializable inside the container)
  and a deterministic **sampler** (greedy / temperature / top-k /
  top-p, seeded xorshift64*). `pai_model_open` / `pai_session_create` /
  `pai_generate` are real: `test_model` builds a tiny LM `.pai`,
  opens it, and generates a deterministic token stream end-to-end —
  the Phase 2 "first generated token" vertical slice on host.
- **KV Cache Manager** (whitepaper §19): per-session KV cache —
  buddy-backed block allocation (§16 GPU allocator foundation),
  rolling FNV-1a prefix hashing for prefix dedup, scheduler
  integration hooks, reserved quantization/spill flags (Phase 3+).
- **Quantization** (whitepaper §15): quantization-agnostic reference
  schemes — symmetric per-group **q8** (int8) and **q4** (packed
  nibbles) with per-group f32 scales, expressed with the same metadata
  the IR carries (bit width / signedness / group size / block
  structure). Packed value + scale layout matches the manifest's
  `size_bytes`; quantized containers dequantize to f32 in the planned
  region at session init (reference execution path).
- **Optimized CPU backend** (whitepaper §6/§12/§37): SIMD operators
  with runtime dispatch (portable → SSE2 → **AVX2+FMA** via cpuid
  detection and function-level target attributes — no extra build
  flags): vecadd / vecmul / scale / relu / softmax / GEMM / GEMV and a
  quantized-weights **W8A16 GEMM** (int8 weights + per-group scales,
  `vpmaddwd`-style dot products). Every op is validated
  differentially against the reference oracle; the ref backend gained
  `pai_ref_gemm_w8_f32` / `pai_ref_gemm_w4_f32` twins.
- **Model importer** (whitepaper §8, first Desktop rung): a text model
  description (values/ops/tokens + raw f32 weight files) is converted
  into a `.pai` container — graph reconstruction → Prospero IR →
  optional quantization → packaging. Gives the §20 `pai convert`
  tooling a real backend.
- **GGUF / LLaMA adapter** (whitepaper §9.3, the first real-model
  compatibility target): `pai convert` auto-detects the GGUF magic.
  - `gguf.{h,c}` — a bounds-checked GGUF v3 reader (header, metadata
    KV with typed getters incl. arrays, tensor infos, alignment) plus
    scalar dequantizers for **F32/F16/BF16/Q4_0..Q8_1 and all K-quants
    (Q2_K..Q8_K)**, ported verbatim from llama.cpp (block layouts from
    `ggml-common.h`, formulas from `ggml-quants.c`).
  - `llama.{h,c}` — translates a LLaMA-2/3 or Mistral GGUF into a
    native `.pai`: architecture metadata, tokenizer vocab + merges,
    transposed weights, and a faithful sequence-mode compute graph
    (RMSNorm, QKV projections, RoPE, causal multi-head attention,
    SiLU-gated MLP, residuals, LM head) with per-position logits as
    the container output. The RoPE cos/sin tables are params excluded
    from quantization.
- **Transformer ops** (whitepaper §12): the reference executor gained
  **SiLU**, **RMSNorm with γ**, **RoPE** (per-position cos/sin table)
  and **causal multi-head attention** (GQA-capable, softmax over the
  causal window), and the model session runs **sequence-mode
  generation** — full-context one-hot input, per-position logits,
  sample the last row (v0 recomputes attention per step instead of a
  KV cache; `kv_bytes_per_token = 0`).
- **`pai` CLI** (whitepaper §20 official tooling): `pai convert`
  (import + quantize), `pai inspect` (sections / meta / manifest / IR /
  tokenizer dump), `pai validate` (CRC + bounds + full model load),
  `pai benchmark` (end-to-end generation timing), `pai serve`
  (OpenAI-compatible gateway, §26; local `.pai` models and/or remote
  payloads via `--remote [name@]host:port`) and
  `pai proto-ping <host> <port>` (Prospero Protocol connectivity
  check — negotiate, then report per-ping RTT with min/median/max and
  loss via `pai_proto_ping`).
- **Benchmark harness** (whitepaper §31): monotonic ns clock + median
  stats; `bench_gemm` validates correctness against the reference
  before timing, then reports min/median GFLOPS for the f32 and w8
  paths on the detected backend.
- **OpenAI-compatible Gateway** (whitepaper §26, the first Desktop
  serving rung): a host-side HTTP/1.1 server speaking the OpenAI wire
  protocol against local `.pai` models **and remote payloads over the
  Prospero Protocol**. `pai serve <model.pai>... [--remote
  [name@]host:port]... [--host H] [--port N]` exposes:
  - `GET /v1/models` — registry listing (and per-model lookup)
  - `POST /v1/completions` and `/v1/chat/completions` — tokenizer +
    sampler parameters (`temperature`/`top_p`/`top_k`/`max_tokens`),
    chat `messages` → prompt assembly, OpenAI-style response envelope
    with `usage` accounting
  - `POST /v1/embeddings` — mean-pooled token embeddings via a new
    `pai_embed` SDK bridge (`pai_model_embed` / `pai_model_embed_dim`)
  - `stream: true` → SSE with per-token `chat.completion.chunk`
    frames and `data: [DONE]`
  Remote entries (`pai_gw_add_remote` / the CLI `--remote` flag) are
  served by **bridging each generation over the protocol**: the
  gateway connects to the payload, negotiates, sends GENERATE, and
  relays each TOKEN chunk straight into the response (`remote.c`,
  `pai_gw_remote_generate`). Unreachable payloads and negotiation
  refusals answer 502 Bad Gateway; remote embeddings are reported as
  501 in v0 (no EMBED message yet), and v0 GENERATE carries only the
  prompt, so remote sampling uses payload defaults.
  Ships its own dependency-free stack: `json.{h,c}` (bounds-checked
  JSON DOM parser with depth/node caps + writer), `gwsys.{h,c}`
  (Win32/POSIX sockets, threads, mutexes), `http.{h,c}` (request
  parser with header/body caps, chunked responses, one-request-per-
  connection keep-alive-free v0). Models are lazily loaded and served
  under a per-model mutex; response ids are registry-unique. Error
  responses follow the OpenAI envelope (`error.message/type/param/
  code`) with 400/404/500/502 semantics, including a friendly 400 for
  prompts containing tokens outside the model's vocabulary.
- **Prospero Protocol** (whitepaper §24/§25): transport-independent
  binary protocol — 40-byte frames with CRC-32, request ids with
  pipelining, version + capability negotiation, sessions, structured
  status codes, native async streams (GENERATE → ACCEPTED → TOKEN* →
  COMPLETE). Two transports plug into the same `pai_proto_transport_t`:
  an in-memory pipe pair (host tests / loopback) and a **TCP
  transport** (the desktop ↔ PS5 link): `pai_proto_tcp_connect`,
  `pai_proto_tcp_listen`/`accept` with ephemeral-port support, and
  pipe-compatible recv semantics (SO_RCVTIMEO-bounded, EOF →
  `PAI_PROTO_CLOSE_PEER_GONE`), plus a `pai_proto_ping` health check
  (negotiate + timed PING/PONG with per-ping timeout and loss
  accounting). `test_tcp` runs the full §24 exchange, pipelined
  PING/PONG, EOF, refused-connect, negotiation-refusal and
  lost-ping paths over real sockets.
- CPU reference backend (correctness oracle): vecadd / vecmul / GEMM /
  relu / softmax / RMSNorm / LayerNorm / concat / copy / memset16
- PM4 command-stream builder (hardware-qualified packet encodings)
- GPU HAL with two backends:
  - `ps5-gc`: raw PM4 submission via `/dev/gc` ioctl 0xC0108102 on real HW
  - `host-ref`: PM4 interpreter executing the same streams on the host
- gfx1013 vecadd compute kernel (llvm-mc assembled)
- `prosperoai.elf` PAI-M0 harness: GPU DMA → compute dispatch → CPU
  reference comparison → benchmark
- **M0-E experiment matrix** (stage E, 9.40 batch diagnostics):
  store_const (E1) / loadstore (E2) control kernels (llvm-mc verified),
  ACO bit-15 FLAT patch variants (E1b/E2b), memset golden
  single-group (E3) and multi-group (E4)
- **Deploy automation**: every payload asks the previous instance to
  terminate itself before starting (TCP stop probe on port 9025, same
  pattern as MemDBG's `--replace-existing`), then binds the port for the
  next deploy. A deployment banner ("ProsperoAI deployed. Credit:
  SeregonWar") is shown on console via `sceKernelSendNotificationRequest`.
- **On-console logging**: the payload escapes the process jail (MemDBG
  pattern: system auth id + full caps + root vnode retarget) and appends
  every log line to `/data/prosperoai/prosperoai.log`, in addition to
  stdout and the kernel log.

## Prerequisites

- **Host builds**: CMake ≥ 3.20, Ninja, Clang (or MSVC)
- **PS5 cross builds**: the OpenOrbis `ps5-payload-sdk` checkout in
  `ps5-payload-sdk/` (not committed; see `.gitignore`) and Clang ≥ 20 on
  the Windows PATH. Note: the vendored `win/prospero-clang.cmd` was adapted
  for Clang 20's native PS5 target support (`SCE_PROSPERO_SDK_DIR` + final
  `--sysroot`); re-apply that patch if the SDK is re-downloaded.
- **Shader assembly** (optional): `llvm-mc`/`llvm-objcopy` (native or WSL)
  and Python 3; otherwise the checked-in prebuilt blobs are used

## Build

```sh
# host build + unit tests
cmake --preset host-tests
cmake --build --preset host-tests
ctest --preset host-tests

# run the M0 harness on the host (reference backends)
./build/host-tests/payload/pai_m0.exe

# PS5 payload (prosperoai.elf)
cmake --preset ps5-debug
cmake --build --preset ps5-debug

# deploy to the console (port 9021)
cmake --build --preset ps5-debug --target pai-deploy
```

## Milestone PAI-M0

**REACHED on physical PS5 hardware (FW 9.40).** The complete bring-up
pipeline — bootstrap, sandbox jailbreak, `/data` logging, deploy
lifecycle, GPU DMA, EOP fence, PM4 submission, compute dispatch,
readback and CPU-reference comparison — executes end-to-end. The
milestone kernel G15 performs **per-thread GPU arithmetic with
runtime user-data** (`c[i] = k + 4i + 3`, k from user data, lanes 0-7)
and passes the CPU-reference check.

The 9.40 silicon has significant undocumented quirks (store data
semantics, vaddr-pair constraint, hardware-zeroed s0-s1, dst-v0
broadcast, 8-lane exec mask, broken float adds, hanging flat loads).
All are documented in `notes/re/940-gpu-empirics.md`; Phase 1 builds
directly on those rules.
