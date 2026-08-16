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
- **Prospero Protocol** (whitepaper §24/§25): transport-independent
  binary protocol — 40-byte frames with CRC-32, request ids with
  pipelining, version + capability negotiation, sessions, structured
  status codes, native async streams (GENERATE → ACCEPTED → TOKEN* →
  COMPLETE). Ships with an in-memory pipe transport for host tests;
  TCP/Local transports plug into the same `pai_proto_transport_t`.
- CPU reference backend (correctness oracle): vecadd / GEMM / memset16
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
