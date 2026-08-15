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
- CPU reference backend (correctness oracle): vecadd / GEMM / memset16
- PM4 command-stream builder (hardware-qualified packet encodings)
- GPU HAL with two backends:
  - `ps5-gc`: raw PM4 submission via `/dev/gc` ioctl 0xC0108102 on real HW
  - `host-ref`: PM4 interpreter executing the same streams on the host
- gfx1013 vecadd compute kernel (llvm-mc assembled)
- `prosperoai.elf` PAI-M0 harness: GPU DMA → compute dispatch → CPU
  reference comparison → benchmark

## Prerequisites

- **Host builds**: CMake ≥ 3.20, Ninja, Clang (or MSVC)
- **PS5 cross builds**: the OpenOrbis `ps5-payload-sdk` checkout in
  `ps5-payload-sdk/` (not committed; see `.gitignore`)
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

Verified GPU tensor compute on physical PS5 hardware. The harness proves,
in order: `/dev/gc` submission (DMA copy), gfx1013 compute dispatch
(vecadd), CPU↔GPU correctness, and dispatch latency.
