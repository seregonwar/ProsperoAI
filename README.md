<div align="center">

<img src="assets/logo_prosperoai.png" width="720" alt="ProsperoAI" />

[![Status: active development](https://img.shields.io/badge/Status-active%20development-f59e0b?style=flat-square)](#project-status)
[![Milestone: PAI-M0/M1](https://img.shields.io/badge/Milestone-PAI--M0%2FM1%20serial-16a34a?style=flat-square)](#project-status)
[![Platforms: PS5 / Host](https://img.shields.io/badge/Platforms-PS5%20%2F%20Host-2f6feb?style=flat-square)](#platform-support)
[![Firmware: 9.40 validated](https://img.shields.io/badge/Firmware-9.40%20validated-0ea5e9?style=flat-square)](#ps5-live-validation)



### Native AI runtime for the jailbroken PlayStation 5.

</div>


**ProsperoAI** is a from-scratch AI runtime that executes neural networks directly
on the PlayStation 5 GPU, with a full reference implementation for desktop
development. It covers the whole stack: raw GPU submission, compute kernels,
model import, quantization, inference, and an OpenAI-compatible serving layer.

It is designed for offline homebrew development, reverse-engineering education,
preservation, and authorized research on consoles you own. It is not a tool for
piracy, account abuse, or unauthorized access.

## Contents

- [Why ProsperoAI](#why-prosperoai)
- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Build Targets](#build-targets)
- [Testing](#testing)
- [PS5 Live Validation](#ps5-live-validation)
- [Protocol](#protocol)
- [Platform Support](#platform-support)
- [Documentation](#documentation)
- [Project Status](#project-status)
- [Responsible Use](#responsible-use)
- [Credits](#credits)

## Why ProsperoAI

ProsperoAI brings the workflows normally split across payloads, GPU drivers,
model converters, and serving stacks into one capability-aware runtime.

| Area | What ProsperoAI provides |
|---|---|
| GPU | Raw PM4 command streams through `/dev/gc` — no GNM/AGC driver, no vendor stack |
| Models | GGUF/LLaMA import into a native `.pai` container with quantization |
| Inference | Compute graph, scheduler, memory planning, reference and SIMD CPU backends |
| Serving | OpenAI-compatible HTTP gateway and a binary TCP protocol for desktop ↔ PS5 |
| Bring-up | PAI-M0 harness with a batched experiment matrix for undocumented silicon |

## Capabilities

### GPU Backend

- PM4 command-stream builder with hardware-qualified packet encodings
- Two HAL backends: `ps5-gc` (raw `/dev/gc` submission on hardware) and
  `host-ref` (PM4 interpreter running the same streams on the host)
- gfx1013 compute kernels (llvm-mc assembled) plus an experiment matrix
  (E/G/F/H batches) to bisect 9.40 dispatch quirks
- Deploy automation: self-terminating previous instances, on-console
  notification banner, and persistent logging in `/data/prosperoai/`

### Runtime

- Compute graph with Kahn topological sort, cycle detection, tensor lifetime
  analysis, and integrated static memory planning
- Prospero IR with binary serialization and CRC-32 integrity (`.pai` container)
- Scheduler with CPU/GPU placement, memory modes, and session registry
- Optimized CPU backend with runtime dispatch (SSE2 / AVX2+FMA) and
  quantized-weights GEMM

### Models

- GGUF v3 reader with F32/F16/BF16/Q4_0..Q8_1 and all K-quant dequantization
- LLaMA-2/3 and Mistral translation to native `.pai` containers
- Byte-level BPE tokenizer, deterministic sampler (greedy / temperature /
  top-k / top-p)
- Transformer ops: RMSNorm, RoPE, causal multi-head attention (GQA),
  SiLU-gated MLP

### Serving

- OpenAI-compatible gateway: `/v1/models`, `/v1/completions`,
  `/v1/chat/completions`, `/v1/embeddings`, SSE streaming with
  full §26 sampling (`stop` sequences, `seed`, `echo`,
  `response_format json_object`, `stream_options.include_usage`)
- Prospero Protocol over TCP with negotiation, pipelining, async streams,
  and ping health checks
- `pai` CLI: `convert`, `inspect` (con `--json`), `optimize`
  (piano mixed-precision §15), `validate`, `benchmark`, `serve`,
  `proto-ping`

## Architecture

```
ProsperoAI
|-- payload/          PAI-M0 harness (prosperoai.elf): stages A / E / B0 / B / C
|-- gpu/
|   |-- hal/          GPU HAL: ps5-gc (raw /dev/gc) + host-ref interpreter
|   |-- pm4/          PM4 command-stream builder
|   `-- kernels/      gfx1013 shaders (llvm-mc assembled) + experiment batches
|-- runtime/          runtime core, public API, version
|-- models/           .pai containers, tokenizer, sampler, importer
|-- adapters/llama/   GGUF reader + LLaMA/Mistral translator
|-- cpu/              reference oracle + SIMD optimized operators
|-- scheduler/        execution planner, placement, memory modes
|-- memory/           static planner + buddy suballocator
|-- graph/ ir/        compute graph + Prospero IR
|-- tensor/ cache/    tensor descriptors, dtypes, KV cache
|-- gateway/          OpenAI-compatible HTTP serving + remote payload bridge
|-- protocol/         Prospero Protocol: frames, TCP/pipe transports
|-- pai/              CLI tooling
|-- platform/         PS5 privilege, lifecycle, notifications, logging
|-- diagnostics/      structured logging
|-- docs/ notes/      whitepaper + RE research
`-- tests/            host test suite (ctest) + seeded protocol fuzzer
```

The payload talks to the desktop through the Prospero Protocol; the gateway
serves local `.pai` models directly and bridges remote generations and
embeddings to PS5 payloads over the same protocol (GENERATE/TOKEN/COMPLETE
with an optional sampler trailer, and the one-shot EMBED/EMBEDDING
exchange).

## Quick Start

### Host Development

The host build runs the same runtime and PM4 streams locally — the fastest
way to develop without hardware.

```sh
cmake --preset host-tests
cmake --build --preset host-tests
ctest --preset host-tests

# run the PAI-M0 harness on the host reference backend
./build/host-tests/payload/pai_m0.exe
```

### PS5 Payload

```sh
cmake --preset ps5-debug
cmake --build --preset ps5-debug

# deploy to the console (port 9021)
cmake --build --preset ps5-debug --target pai-deploy
```

Every deploy asks the previous instance to terminate itself (TCP stop probe),
binds the port for the next deploy, and shows a notification banner on the
console. Logs are appended to `/data/prosperoai/prosperoai.log`.

## Build Targets

### Requirements

| Component | Requirement |
|---|---|
| Host builds | CMake ≥ 3.20, Ninja, Clang (or MSVC) |
| PS5 payload | OpenOrbis `ps5-payload-sdk` in `ps5-payload-sdk/` (not committed), Clang ≥ 20 |
| Shader assembly (optional) | `llvm-mc` / `llvm-objcopy`, Python 3; prebuilt blobs are checked in |

### Presets

| Preset | Purpose |
|---|---|
| `host-tests` | host debug build + unit test suite |
| `host-reference` | host release build, no tests |
| `host-sanitized` | ASan + UBSan build with tests |
| `ps5-debug` | PS5 payload with verbose diagnostics |
| `ps5-release` | PS5 payload, optimized |
| `ps5-safe` | CPU-only diagnostics, no GPU submission |

## Testing

```sh
ctest --preset host-tests        # full host suite
ctest --preset host-sanitized    # ASan + UBSan run
```

The suite covers tensors, memory allocation, PM4 encoding, reference ops,
the host GPU interpreter, protocol, scheduler, models, quantization,
importers, and the gateway — including real-socket TCP protocol tests.

## PS5 Live Validation

The complete bring-up pipeline was validated on physical PS5 hardware
(FW 9.40):

| Check | Result |
|---|---|
| Sandbox jailbreak + `/data` logging | verified |
| Deploy lifecycle (stop probe, re-deploy) | verified |
| GPU DMA copy (1 MiB) | verified vs CPU reference |
| EOP fence (action / write-data / legacy ladder) | verified |
| Compute dispatch (golden memset16) | verified |
| Per-thread GPU arithmetic with user data (G15) | verified vs CPU reference |

The 9.40 silicon has undocumented quirks (store data semantics, vaddr-pair
constraint, hardware-zeroed s0-s1, dst-v0 broadcast, 8-lane exec mask). All
are catalogued in `notes/re/940-gpu-empirics.md`; Phase 1 builds directly on
those rules.

## Protocol

The Prospero Protocol is a compact binary protocol for the desktop ↔ PS5
link: 40-byte frames with CRC-32, version/capability negotiation, request
pipelining, sessions, structured status codes, and native async streams
(GENERATE → ACCEPTED → TOKEN* → COMPLETE). Two transports plug into the same
interface: an in-memory pipe pair (host tests) and TCP. The normative spec
lives in the whitepaper (§24/§25).

## Platform Support

| Target | Status | Notes |
|---|---|---|
| PlayStation 5 | Supported | FW 9.40 validated; raw `/dev/gc` submission, no driver |
| Host (Windows / Linux) | Supported | reference interpreter + full runtime |

## Documentation

| Document | Purpose |
|---|---|
| [`docs/ProsperoAI - Technical Whitepaper v0.1.md`](docs/ProsperoAI%20-%20Technical%20Whitepaper%20v0.1.md) | Full architecture (whitepaper §1–§43) |
| [`notes/re/940-gc-ioctl.md`](notes/re/940-gc-ioctl.md) | `/dev/gc` ioctl ABI research |
| [`notes/re/940-gpu-empirics.md`](notes/re/940-gpu-empirics.md) | Undocumented 9.40 GPU behavior |

## Project Status

**Milestones PAI-M0 and PAI-M1 (serial path) reached on physical PS5
hardware (FW 9.40).** Bring-up covers DMA, EOP fence, PM4 compute,
scalar `s_load`, integer SAXPY, serial reduction, and serial-per-row
GEMV — all CPU-reference-checked on the G22 path. Parallel reduction
(M1C / LDS) remains blocked pending a real AGC CS LDS blob
(`COMPUTE_PGM_RSRC2` @ SH `0x213`); see `notes/re/940-gpu-empirics.md`.

Whitepaper Phase 1 (full tensor runtime) and Phase 2 (first token) still
lie ahead. The host reference path already runs end-to-end model
generation (`test_model`, `pai serve`).

## Responsible Use

ProsperoAI is built for legitimate homebrew development, reverse-engineering
education, preservation, and security research on systems you own or are
authorized to analyze.

The maintainers do not support piracy, account abuse, unauthorized access to
systems or services, or disruptive behavior against online services. This
project is provided as-is; the authors are not responsible for misuse,
hardware or software damage, or violations of third-party terms.

## Credits

- Developed by SeregonWar
- GPU bring-up based on reverse engineering from the PS5 scene (OpenAGC)

---

**ProsperoAI** - high-performance AI runtime for PS5 research environments.
