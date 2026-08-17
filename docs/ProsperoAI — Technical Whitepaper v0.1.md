# ProsperoAI

## High-Performance General-Purpose AI Runtime for Jailbroken PlayStation 5

**Status:** Architecture Draft  
**Version:** 0.1  
**Primary target:** PlayStation 5 homebrew environments  
**Core implementation:** ISO C + platform-specific intrinsics/assembly where justified  
**Desktop toolchain:** Electron + TypeScript + Rust  
**Native model format:** `.pai` — ProsperoAI Model Container

---

## 1. Abstract

ProsperoAI is a high-performance, general-purpose artificial intelligence runtime designed specifically for jailbroken PlayStation 5 systems.

The project aims to transform the PS5 into a capable local AI inference platform by exploiting its CPU, GPU and unified memory architecture through a purpose-built runtime rather than relying exclusively on frameworks originally designed for desktop operating systems or discrete PC GPUs.

The primary design objective is **maximum achievable inference performance on PS5 hardware**.

ProsperoAI follows a GPU-first architecture. Model weights, execution state and computationally intensive tensor workloads should remain GPU-resident whenever possible. CPU resources are reserved for operations that measurably benefit from CPU execution, including scheduling, tokenization, sampling, control-flow-heavy workloads, selected preprocessing operations and system management.

The project is not intended to be limited to large language models. LLM inference represents the first practical implementation target, but the runtime, intermediate representations, model format and execution engine are designed from the beginning to accommodate additional workloads including:

- embedding models;
- rerankers;
- vision transformers;
- image processing;
- speech recognition;
- speech synthesis;
- diffusion models;
- multimodal architectures;
- future tensor-based workloads.

ProsperoAI consists of two major components:

1. **ProsperoAI Payload**, a single modular PS5 payload written primarily in C;
2. **ProsperoAI Desktop**, a cross-platform model management and compilation environment built with Electron, TypeScript and Rust.

The Desktop component performs model discovery, conversion, quantization, preprocessing, compilation, optimization, transfer and management. The PS5 component concentrates on execution, hardware-specific optimization, profiling and runtime scheduling.

---

# 2. Project Vision

ProsperoAI is not intended to be a simple port of `llama.cpp`.

Nor is it merely a PS5 frontend for an existing inference runtime.

Existing runtimes may be supported through adapter layers, but the long-term architecture is independent from them.

The intended stack is:

```text
Applications / IDEs / Homebrew / Clients
                    │
          ┌─────────┴─────────┐
          │                   │
 ProsperoAI SDK       OpenAI-Compatible API
          │                   │
          └─────────┬─────────┘
                    │
             Prospero Protocol
                    │
            ProsperoAI Runtime
                    │
        ┌───────────┼────────────┐
        │           │            │
   Native Engine   Adapters   Model Manager
        │
        ▼
    Prospero IR
        │
        ▼
 Graph Compiler / Optimizer
        │
        ▼
    Execution Planner
        │
  ┌─────┴─────────────┐
  │                   │
GPU Backend        CPU Backend
  │                   │
  └─────────┬─────────┘
            ▼
       PS5 Hardware
```

The design philosophy is:

> Existing AI runtimes adapt themselves to the PS5.  
> ProsperoAI adapts AI workloads to the PS5.

---

# 3. Core Design Principles

## 3.1 Performance First

ProsperoAI prioritizes performance over simplicity of implementation.

Where meaningful performance improvements require:

- platform-specific code;
- custom memory management;
- kernel specialization;
- runtime code generation;
- kernel patches;
- hardware-specific execution plans;
- model-specific layouts;

ProsperoAI should prefer those approaches over generic abstractions.

Portability must not impose significant overhead on the PS5 execution path.

---

## 3.2 GPU-First Execution

The PS5 GPU is considered the primary computational device.

The preferred execution model is:

```text
Model Weights ──────────────┐
KV / Runtime State ─────────┤
Intermediate Tensors ───────┤──► GPU resident
Large Compute Operations ───┘

CPU
 ├── scheduler
 ├── tokenizer
 ├── sampler
 ├── protocol
 ├── I/O coordination
 ├── graph/runtime management
 └── operators proven faster on CPU
```

CPU execution is therefore not a fallback merely because an implementation is easier to write.

An operator should execute on CPU only when profiling indicates that doing so is advantageous or necessary.

---

## 3.3 General-Purpose Tensor Architecture

ProsperoAI must avoid assumptions such as:

- every model is autoregressive;
- every graph contains attention;
- every workload generates tokens;
- every model is transformer-based.

LLMs are the first target, not the architectural boundary.

The tensor engine therefore operates on generic graphs, tensors, operators, memory dependencies and execution plans.

---

## 3.4 Hardware-Aware Compilation

A model should not be treated as a static sequence of generic operators.

ProsperoAI analyzes:

- model architecture;
- tensor shapes;
- quantization;
- memory requirements;
- available GPU resources;
- firmware capabilities;
- benchmark results;
- workload characteristics;

and generates an execution strategy specifically optimized for the target console.

---

# 4. Target Environment

ProsperoAI targets jailbroken PlayStation 5 systems with varying levels of low-level access.

Kernel patching can generally be assumed to be available in supported environments.

Hypervisor control, however, cannot.

The architecture must therefore never treat an unlocked hypervisor as a mandatory baseline dependency.

Features are exposed through capability detection:

```text
Capability available?
        │
   ┌────┴────┐
  yes        no
   │          │
optimized   fallback
 path        path
```

Potential HV-dependent optimizations should remain isolated behind dedicated capability flags.

A console without those capabilities must still be capable of running ProsperoAI whenever the underlying GPU and memory interfaces required by the baseline backend remain accessible.

---

# 5. Single-Payload Architecture

ProsperoAI will be deployed on the console as a **single payload**.

The deployment model intentionally avoids requiring a collection of independently installed services.

Conceptually:

```text
prosperoai.elf
│
├── Bootstrap
├── Platform Detection
├── Kernel Patch Manager
├── Capability Manager
│
├── Memory Subsystem
├── GPU HAL
├── GPU Runtime
├── CPU Backend
│
├── Tensor Engine
├── Prospero IR
├── Graph Compiler
├── Kernel Compiler
├── JIT
│
├── Scheduler
├── KV Cache Manager
├── Model Manager
├── Runtime Adapters
│
├── Model Repository
├── Profiler
├── Autotuner
│
├── Prospero Protocol
├── OpenAI Gateway
├── Security Layer
├── Diagnostics
└── Recovery System
```

These components remain logically isolated despite being compiled into one executable.

This gives ProsperoAI the deployment simplicity of a single payload without forcing the source architecture to become monolithic.

---

# 6. Implementation Language

The PS5 runtime will be implemented primarily in **C**.

C is selected because it provides:

- predictable ABI;
- minimal runtime requirements;
- straightforward integration with PS5 homebrew toolchains;
- precise control over memory;
- easy interaction with kernel interfaces;
- direct access to intrinsics;
- easy integration with architecture-specific assembly;
- little hidden execution overhead.

Performance-critical CPU routines may use:

- AVX2;
- FMA;
- architecture-specific intrinsics;
- hand-written x86-64 assembly.

The project therefore defines "C implementation" as the architectural language choice, not as a prohibition against using specialized assembly where profiling justifies it.

---

# 7. ProsperoAI Desktop

The second major component is **ProsperoAI Desktop**.

Technology stack:

```text
UI                TypeScript
Desktop shell     Electron
Toolchain backend Rust
```

Desktop is considerably more than a file transfer utility.

It acts as the host-side compiler and model preparation environment for ProsperoAI.

Its responsibilities include:

- discovering models;
- integration with Hugging Face;
- downloading models;
- local model imports;
- URL imports;
- custom repositories;
- ProsperoAI Registry access;
- validation;
- architecture detection;
- conversion;
- quantization;
- mixed-precision planning;
- graph analysis;
- Prospero IR generation;
- AOT optimization;
- `.pai` creation;
- candidate kernel generation;
- model transfer;
- PS5 management;
- runtime update management;
- profiling visualization;
- benchmark visualization;
- diagnostic analysis.

The default UX should aim for:

```text
Search
  ↓
Select model
  ↓
Install
  ↓
Run
```

An **Expert Mode** exposes the underlying controls.

---

# 8. Model Import Pipeline

For native high-performance execution, heavyweight model parsing and conversion should occur on the desktop rather than the PS5.

Example Hugging Face pipeline:

```text
Hugging Face Model
        │
        ├── safetensors
        ├── configuration
        ├── tokenizer
        └── architecture metadata
        │
        ▼
Model Importer
        │
        ▼
Graph Reconstruction
        │
        ▼
Prospero IR
        │
        ▼
Graph Optimization
        │
        ▼
Quantization Planner
        │
        ▼
Kernel Candidate Generation
        │
        ▼
Memory Planning
        │
        ▼
PAI Packaging
```

GGUF will also be supported, initially providing a practical route for compatibility with the existing LLM ecosystem.

---

# 9. Runtime Compatibility Layers

ProsperoAI supports external runtimes using three integration levels.

## 9.1 Passthrough

The external runtime remains responsible for most execution.

ProsperoAI mainly provides platform services.

This prioritizes compatibility.

---

## 9.2 Accelerated

The external runtime delegates selected operations to ProsperoAI:

- memory allocation;
- GPU execution;
- optimized operators;
- tensor transfers;
- profiling.

---

## 9.3 Native Translation

Where possible, the external runtime graph is translated into Prospero IR.

The model can then use the complete native pipeline:

```text
External Runtime
       ↓
Graph Adapter
       ↓
Prospero IR
       ↓
Prospero Compiler
       ↓
Native PS5 Execution
```

This allows a newly supported runtime to start with passthrough compatibility and progressively acquire native performance.

`llama.cpp`/GGUF will be the first major compatibility target.

---

# 10. Prospero Intermediate Representation

ProsperoAI uses at least two conceptual IR levels.

## 10.1 Prospero IR

Represents the computational model at graph level.

It describes:

- tensors;
- shapes;
- datatypes;
- quantization metadata;
- operators;
- dependencies;
- control flow;
- memory properties;
- device placement constraints.

Prospero IR remains hardware-aware enough to allow optimization, but not PS5-kernel-specific.

---

## 10.2 Kernel IR

After graph optimization and fusion, sections of Prospero IR are lowered into a lower-level Kernel IR.

```text
Prospero IR
     ↓
Graph passes
     ↓
Fusion
     ↓
Kernel IR
     ↓
Kernel generation
```

Kernel IR describes computational units close enough to GPU execution to permit:

- tiling;
- vector width selection;
- memory layout specialization;
- workgroup configuration;
- quantized packing;
- operator fusion;
- shape specialization.

---

# 11. Graph Compiler

Native `.pai` execution should primarily follow a compiled graph model rather than purely eager execution.

Pipeline:

```text
Model
 ↓
Prospero IR
 ↓
Canonicalization
 ↓
Constant folding
 ↓
Layout optimization
 ↓
Operator fusion
 ↓
Device placement
 ↓
Kernel selection
 ↓
Memory planning
 ↓
Execution scheduling
 ↓
Execution Plan
```

Dynamic/eager execution remains available for:

- unsupported operators;
- runtime adapters;
- dynamic graphs;
- debugging;
- compatibility modes.

---

# 12. GPU Kernel Architecture

ProsperoAI will combine three kernel strategies.

## Hand-Tuned Kernels

Critical operators receive manually optimized implementations.

Initial priorities include:

- GEMM;
- GEMV;
- quantized matrix multiplication;
- attention;
- normalization;
- activation functions;
- softmax;
- rotary embeddings;
- tensor conversion.

---

## Generated Kernels

ProsperoAI can generate kernels from Kernel IR for broader operator coverage.

---

## Parameterized Kernels

Single kernel families may expose variables such as:

- tile dimensions;
- workgroup dimensions;
- vector width;
- tensor layout;
- packing strategy;
- prefetch depth;
- shared/local memory use.

This creates many candidate implementations without maintaining each as separate source code.

---

# 13. Autotuning

Autotuning is a first-class feature.

ProsperoAI does not assume that a theoretically ideal kernel configuration is optimal on PS5.

Instead:

```text
Kernel family
     ↓
Generate candidates
     ↓
Benchmark on PS5
     ↓
Validate correctness
     ↓
Compare performance
     ↓
Store winner
```

Tuning may evaluate:

- tile dimensions;
- memory layouts;
- fusion strategies;
- dispatch sizes;
- CPU/GPU placement;
- quantization layouts;
- cache strategies;
- prefetch distances.

The selected configuration is persisted in an **Execution Profile**.

Subsequent runs avoid repeating unnecessary tuning.

---

# 14. AOT and JIT Compilation

ProsperoAI uses a hybrid compilation model.

### Desktop/AOT

ProsperoAI Desktop performs expensive analysis and generates candidate implementations before deployment.

### Console specialization

The PS5 specializes the model against real hardware capabilities.

### Selective JIT

Runtime JIT is permitted when specialization can provide measurable gains.

Compiled results are cached.

A `no-jit` mode remains available for:

- debugging;
- deterministic testing;
- safe mode;
- compatibility investigation.

---

# 15. Quantization Architecture

ProsperoAI is **quantization-agnostic**.

The runtime must not be designed around a small fixed list such as INT8 and INT4.

Instead, quantization is represented through metadata describing:

- bit width;
- group size;
- scale representation;
- zero-point representation;
- block structure;
- packing;
- signedness;
- layout;
- per-tensor parameters;
- per-channel parameters;
- per-layer parameters.

The Desktop toolchain can choose different quantizations for different portions of a model.

Example:

```text
Embedding        FP16
Attention        Q6
MLP              Q4
Output layer     Q8
KV cache         Q8 / Q4
```

The choice should eventually be driven by:

- memory constraints;
- accuracy requirements;
- hardware capabilities;
- model architecture;
- measured PS5 performance.

---

# 16. Memory Architecture

Memory management is considered as important as compute optimization.

ProsperoAI provides several complementary systems.

## Static Memory Planner

Compiled graphs allow tensor lifetimes to be determined ahead of execution.

Buffers can therefore be reused when lifetimes do not overlap.

---

## Arena and Pool Allocators

Dynamic runtime state uses specialized pools rather than depending heavily on generic heap allocation.

Separate arenas may exist for:

- models;
- sessions;
- KV cache;
- temporary tensors;
- network buffers;
- profiling;
- compiler/JIT state.

---

## GPU Allocator

A dedicated GPU allocator manages large memory regions and performs suballocation.

Objectives include:

- reducing fragmentation;
- predictable alignment;
- fast allocation;
- efficient reuse;
- minimizing unnecessary copies.

---

# 17. Execution Memory Modes

ProsperoAI exposes three primary memory profiles.

## Performance

The complete model is kept GPU-resident whenever possible.

Goal:

**maximum throughput and minimum latency.**

---

## Balanced

GPU residency is preferred while selected model data may remain in system memory.

The scheduler determines which tensors benefit most from GPU residency.

---

## Capacity

Designed for models larger than immediately available memory.

ProsperoAI uses a hierarchical virtual tensor memory system:

```text
        Hot data
           │
           ▼
      GPU Memory
           ↕
        RAM Cache
           ↕
      SSD Storage
           │
           ▼
        Cold data
```

The runtime manages:

- asynchronous prefetch;
- eviction;
- double/triple buffering;
- staging buffers;
- access prediction;
- tensor lifetime;
- bandwidth measurements;
- graph-aware scheduling.

The intent is to hide as much storage latency as possible behind GPU execution.

---

# 18. Scheduler

ProsperoAI uses a global workload-aware scheduler.

Responsibilities include:

- CPU/GPU placement;
- model residency;
- session priority;
- batch construction;
- memory pressure;
- transfer scheduling;
- asynchronous I/O;
- prefetch;
- kernel dispatch;
- cache management.

For LLMs it supports **continuous batching**.

Prefill and decode are considered fundamentally different workloads and may be scheduled independently.

Available policies should include:

### Interactive

Prioritize low response latency.

### Throughput

Prioritize aggregate tokens/sec across multiple sessions.

### Exclusive Performance

Dedicate the vast majority of available resources to a single workload.

### Balanced

Automatically trade latency, concurrency and memory pressure.

---

# 19. KV Cache Manager

LLM support includes a dedicated KV Cache Manager.

Features should include:

- per-session cache;
- shared prefix caching;
- prefix deduplication;
- cache quantization;
- intelligent eviction;
- optional spill to RAM;
- optional storage spill;
- persistence;
- scheduler integration.

KV cache residency becomes part of global scheduling rather than being independently managed by each inference session.

---

# 20. `.pai` Model Format

`.pai` is the native ProsperoAI model container.

It is both:

1. a portable model package;
2. a vehicle for hardware-specific compiled assets.

The format is publicly specifiable and strongly versioned even while the overall project remains private during early development.

Conceptually:

```text
PAI Header
│
├── Metadata
├── Model Manifest
├── Tokenizer / Preprocessor
├── Canonical Weights
├── Prospero IR
│
├── Target Slice: PS5
│   ├── packed tensors
│   ├── optimized weights
│   ├── kernel candidates
│   └── execution metadata
│
├── Optional Target Slice ...
│
├── Execution Profiles
├── Integrity Data
└── Optional Signature
```

The format supports:

- versioning;
- feature flags;
- optional sections;
- checksums;
- hashes;
- explicit alignment;
- multiple hardware slices;
- multiple execution profiles;
- backward-compatible readers where feasible.

Official tooling will eventually include:

```text
pai inspect
pai validate
pai convert
pai optimize
pai benchmark
```

---

# 21. Universal Container Model

`.pai` is not permanently tied to PS5.

A container can include:

- canonical model representation;
- generic Prospero IR;
- shared metadata;
- PS5-specific target slices;
- future hardware-specific slices.

This makes `.pai` conceptually similar to a fat binary.

Native ProsperoAI execution on PS5 still uses the PS5 slice whenever one exists.

Portability must not compromise the optimized target representation.

---

# 22. Model Repository

ProsperoAI manages installed models through a dedicated repository.

The repository combines conventional files with content-addressed storage.

A `.pai` remains a standalone import/export artifact.

When installed, however, model data may be decomposed into content-addressed blocks.

Advantages include:

- deduplication;
- partial updates;
- transfer resumption;
- model version sharing;
- integrity verification.

---

# 23. Model Transfer Engine

ProsperoAI Desktop uses a dedicated bulk transfer subsystem.

Requirements:

- chunked transfer;
- parallel transfers;
- per-chunk hashes;
- resumability;
- connection-loss recovery;
- final integrity verification;
- deduplication;
- bandwidth limiting;
- transfer priority.

Large model installations must never require restarting from zero after a temporary interruption.

---

# 24. Prospero Protocol

ProsperoAI uses a custom binary protocol influenced by proven design concepts from MemDBG, while remaining a distinct wire protocol with its own ABI and semantics.

MemDBG itself remains untouched.

The new protocol extracts and generalizes useful ideas rather than creating a runtime dependency between the projects.

Important features include:

- compact binary frames;
- request identifiers;
- capability negotiation;
- sessions;
- version negotiation;
- persistent connections;
- pipelining;
- connection roles;
- structured status codes;
- compression where beneficial.

The protocol adds native support for asynchronous streams.

Example:

```text
CLIENT                         PS5

GENERATE #42 ─────────────────►
              ◄──────── ACCEPTED #42

              ◄──────── TOKEN #42
              ◄──────── TOKEN #42
              ◄──────── TOKEN #42
              ◄──────── TOKEN #42

              ◄──────── COMPLETE #42
```

Streaming semantics are also used for:

- telemetry;
- profiler output;
- autotuning;
- transfer progress;
- logs;
- long-running compilation operations.

---

# 25. Transport Layer

The protocol must remain independent of the underlying transport.

Initial transports:

```text
TCP
Local/Internal
```

TCP is the baseline desktop-to-console mechanism.

Bidirectional USB connectivity is not considered a baseline requirement because reliable general-purpose PS5↔PC USB transport may not be practical in target environments.

The abstraction nevertheless allows additional transports to be added later without redesigning the application protocol.

---

# 26. OpenAI-Compatible Gateway

ProsperoAI provides an OpenAI-compatible network layer independently from the native runtime.

The gateway translates external requests into Prospero Protocol/runtime operations.

Initial compatibility should cover at minimum:

- model listing;
- chat completions;
- completions;
- embeddings;
- streamed generation.

Long-term support may include:

- structured outputs;
- tool calling;
- multimodal inputs;
- usage reporting;
- batching extensions.

This allows existing applications to treat a PS5 as a local AI endpoint without requiring native ProsperoAI support.

---

# 27. Developer SDK

ProsperoAI exposes a two-level C SDK.

## High-Level API

Designed for common homebrew applications.

Conceptually:

```c
pai_model_open(...);
pai_model_close(...);

pai_session_create(...);
pai_session_destroy(...);

pai_generate(...);
pai_embed(...);
```

---

## Expert API

Provides lower-level access to:

- tensors;
- graphs;
- Prospero IR;
- memory hints;
- scheduling;
- backend selection;
- profiling;
- compilation;
- execution plans;
- custom operators.

ABI stability should be stronger for the high-level API than for the Expert API during early development.

---

# 28. Firmware and Hardware Compatibility

Hardware and firmware handling is capability-driven.

ProsperoAI combines:

- known firmware profiles;
- signature scanning;
- runtime probing;
- cached symbol/offset resolution;
- dependency-managed kernel patches;
- validation before patching.

Hard-coded firmware offsets should not become the primary architecture.

Each kernel patch has explicit dependencies.

For example:

```text
GPU_MAPPING_PATCH
   requires:
      SYMBOL_A
      SYMBOL_B
      CAP_KERNEL_RW

FAST_PATH_X
   requires:
      GPU_MAPPING_PATCH
      CAP_FEATURE_Y
```

If an optional dependency is unavailable, ProsperoAI should disable the optimization rather than failing the entire runtime where possible.

---

# 29. Capability Fingerprint

Each console generates a capability fingerprint describing relevant characteristics such as:

- firmware;
- hardware revision;
- available kernel primitives;
- GPU backend features;
- memory capabilities;
- HV-dependent capabilities;
- compiler features.

Execution profiles and benchmark results are associated with this fingerprint.

A profile generated for incompatible capabilities must never be blindly reused.

---

# 30. Profiling

ProsperoAI includes a full profiling subsystem.

Metrics include:

- total inference latency;
- time-to-first-token;
- tokens/sec;
- prefill throughput;
- decode latency;
- individual operator latency;
- GPU queue utilization;
- CPU/GPU overlap;
- memory usage;
- transfer bandwidth;
- cache pressure;
- model load time.

Profiling data feeds directly into optimization.

The profiler is therefore not merely a diagnostic UI.

It is part of the runtime feedback loop.

---

# 31. Benchmarking

Benchmarking must be reproducible.

Every official benchmark records:

- ProsperoAI version;
- commit;
- firmware;
- capability fingerprint;
- model;
- `.pai` revision;
- quantization;
- execution profile;
- context length;
- batch configuration;
- memory mode;
- warm/cold state.

Tests run multiple iterations and produce statistical summaries.

Correctness is validated before performance results are accepted.

ProsperoAI Desktop can compare benchmarks between versions and highlight regressions automatically.

---

# 32. Reliability and Recovery

Low-level GPU control and kernel interaction make recovery a fundamental requirement.

ProsperoAI includes:

- internal watchdog;
- structured error domains;
- model transfer journal;
- repository journal;
- crash diagnostics;
- execution-profile validation;
- corrupted-cache invalidation;
- safe mode.

Safe mode may disable:

- JIT;
- aggressive kernel fusion;
- experimental GPU paths;
- autotuning;
- optional kernel patches.

The goal is to allow recovery without requiring deletion of installed models or configuration.

---

# 33. Diagnostics

Diagnostics include:

- structured logs;
- severity levels;
- subsystem identifiers;
- request/session correlation;
- high-resolution timestamps;
- trace spans;
- in-memory ring buffer;
- live Desktop streaming.

ProsperoAI Desktop can generate a **Diagnostic Bundle** containing:

- logs;
- runtime configuration;
- capability fingerprint;
- firmware profile;
- active execution profiles;
- benchmark metadata;
- crash information.

---

# 34. Security

Remote administration is protected by default.

ProsperoAI supports:

- device pairing;
- persistent client identity;
- encrypted communications;
- authorized-client management;
- integrity verification.

A development mode may relax selected restrictions to simplify:

- debugging;
- packet inspection;
- benchmarking;
- early bring-up.

Security must remain configurable because ProsperoAI operates primarily in research/homebrew environments, while unsafe remote defaults should still be avoided.

---

# 35. Payload Updates

ProsperoAI Desktop manages payload updates.

The update architecture supports:

- version negotiation;
- integrity verification;
- signed releases;
- dual-slot deployment;
- rollback;
- differential updates where practical;
- version pinning.

Channels may include:

```text
stable
beta
nightly
```

Firmware-sensitive environments can remain pinned to known-good versions.

---

# 36. Build System

The C codebase uses:

**CMake + Ninja**

with dedicated presets/toolchains.

Example configurations:

```text
ps5-release
ps5-debug
ps5-safe
host-reference
host-tests
host-sanitized
```

Host builds provide a critical development environment even for functionality ultimately intended for PS5.

---

# 37. Reference Backend

A deliberately straightforward C reference backend acts as the correctness oracle.

It is not intended to achieve maximum performance.

Optimized implementations are validated against it.

Testing includes:

- deterministic tensor tests;
- randomized tensors;
- edge cases;
- quantization-specific tolerances;
- differential testing;
- regression vectors.

External frameworks may also be used as secondary references.

---

# 38. CI and Hardware-in-the-Loop Testing

The long-term CI architecture includes:

```text
Host CI
├── build
├── unit tests
├── integration tests
├── sanitizer
├── fuzzing
├── model parser tests
└── protocol tests

Cross Compilation
└── PS5 build validation

Private PS5 Runner
├── deployment
├── kernel tests
├── GPU correctness
├── inference tests
├── protocol integration
├── autotuning tests
└── performance regression
```

Performance-sensitive changes can be rejected when they produce statistically significant regressions beyond configured thresholds.

---

# 39. Initial Repository Structure

```text
ProsperoAI/
│
├── payload/
│
├── platform/
│   └── ps5/
│
├── runtime/
├── tensor/
├── graph/
├── ir/
│
├── gpu/
│   ├── hal/
│   ├── kernels/
│   ├── compiler/
│   └── autotune/
│
├── cpu/
│   ├── reference/
│   └── optimized/
│
├── memory/
├── scheduler/
├── cache/
│
├── models/
├── pai/
├── adapters/
│   └── llama/
│
├── protocol/
├── gateway/
├── sdk/
│
├── profiler/
├── diagnostics/
│
├── desktop/
│   ├── frontend/
│   └── rust/
│
├── toolchain/
├── registry/
│
├── tests/
├── benchmarks/
├── cmake/
└── docs/
```

ProsperoAI remains a monorepo during its initial development.

Individual components may be extracted later if clear independent use cases emerge.

---

# 40. Development Roadmap

ProsperoAI development should progress through measurable milestones rather than attempting to implement the entire architecture before first inference.

## Phase 0 — Hardware Bring-Up

Goal:

**prove reliable compute access to the PS5 GPU.**

Deliverables:

- single payload bootstrap;
- firmware/capability detection;
- GPU initialization;
- GPU memory allocation;
- command submission;
- synchronization;
- basic compute kernel;
- CPU↔GPU correctness test.

A basic matrix/vector operation is sufficient.

This phase is the most important feasibility checkpoint.

---

## Phase 1 — Tensor Runtime

Detailed design: [`phase-1-tensor-runtime.md`](phase-1-tensor-runtime.md).

Implement:

- tensor descriptors;
- datatypes;
- memory planner foundation;
- reference CPU backend;
- GPU backend;
- first optimized operators;
- profiling.

Target:

run a small synthetic neural network completely through ProsperoAI.

---

## Phase 2 — LLM Minimum Viable Runtime

Implement the minimum operators required for a small decoder transformer.

Target:

**first generated token on PS5.**

Performance is measured, but correctness is the primary requirement.

---

## Phase 3 — 1–3B GPU-Resident Model

Target a practical 1–3B parameter LLM.

Requirements:

- stable GPU residency;
- quantized execution;
- KV cache;
- tokenizer;
- sampling;
- streaming generation;
- Prospero Protocol;
- Desktop deployment.

This becomes the first complete end-to-end ProsperoAI system.

---

## Phase 4 — Native `.pai`

Implement:

- format specification;
- Desktop conversion;
- native graph compilation;
- Prospero IR;
- PS5 target slices;
- execution profiles.

GGUF remains supported through adapters.

---

## Phase 5 — Autotuning

Implement:

- kernel candidate generation;
- benchmark harness;
- execution-profile persistence;
- kernel cache;
- regression detection.

---

## Phase 6 — 7–8B Target

The first significant public-quality performance milestone is a practical quantized 7–8B model running locally on PS5.

Success metrics include:

- stable extended inference;
- acceptable time-to-first-token;
- competitive decode throughput for the hardware;
- complete GPU residency where feasible;
- reliable Desktop deployment.

---

## Phase 7 — Balanced and Capacity

Add:

- RAM offload;
- virtual tensor memory;
- SSD streaming;
- prefetch;
- model sizes exceeding directly resident memory.

---

## Phase 8 — Multi-Session Runtime

Add:

- continuous batching;
- prefix caching;
- concurrent sessions;
- global scheduler;
- throughput profiles.

---

## Phase 9 — General AI Runtime

Expand beyond LLMs.

Initial candidates:

- embeddings;
- rerankers;
- vision encoders;
- speech recognition.

The architectural success criterion is that adding them does **not** require redesigning the tensor engine or `.pai`.

---

# 41. Primary Technical Risks

ProsperoAI contains several major engineering uncertainties.

## GPU Interface

The most significant early risk is obtaining sufficiently reliable and controllable access to GPU compute, memory management, command submission and synchronization in the target homebrew environment.

This must be validated before investing heavily in upper runtime layers.

---

## GPU Compiler Path

ProsperoAI must determine the most practical route from generated Kernel IR to executable GPU workloads.

Depending on available interfaces, this may require substantial platform research.

The kernel compiler should therefore be architecturally isolated from the graph compiler.

---

## Memory Availability

Although PS5 possesses substantial unified system memory, the amount practically available to a homebrew payload and GPU workload can depend on the execution environment.

ProsperoAI must measure rather than assume available capacity.

---

## Firmware Differences

Kernel and GPU behavior may vary across firmware families.

Capability detection and patch validation are therefore mandatory rather than optional conveniences.

---

## Hypervisor Restrictions

Some environments retain a locked hypervisor.

ProsperoAI must not require hypervisor control for its baseline execution path.

HV-dependent functionality is considered an optional acceleration tier.

---

# 42. Success Criteria

ProsperoAI should be considered technically successful when it can:

1. execute tensor compute reliably on the PS5 GPU;
2. load and execute a real quantized LLM;
3. keep performance-critical model state GPU-resident;
4. generate stable streamed output;
5. be controlled through ProsperoAI Desktop;
6. expose a usable OpenAI-compatible API;
7. automatically optimize execution for the target console;
8. reproduce performance through persistent execution profiles;
9. support models beyond the original LLM architecture without redesigning the core.

The ultimate objective is not merely:

> “An AI model can run on PS5.”

The objective is:

> **“ProsperoAI turns the PS5 into a purpose-built local AI inference platform and extracts as much useful compute performance from the hardware as the environment allows.”**

---

# 43. Immediate Engineering Priority

The next implementation work should deliberately ignore most high-level features.

The first vertical slice should be:

```text
prosperoai.elf
       ↓
PS5 bootstrap
       ↓
capability detection
       ↓
GPU memory allocation
       ↓
compute submission
       ↓
simple tensor kernel
       ↓
readback
       ↓
CPU reference comparison
       ↓
benchmark
```

Only after this path is reliable should development proceed toward the full model runtime.

The recommended first internal milestone is therefore:

> **PAI-M0: verified GPU tensor compute on physical PS5 hardware.**

Everything else in ProsperoAI ultimately depends on this layer.

### Internal bring-up status (FW 9.40)

**PAI-M0 — COMPLETE.** Physical PS5 validation covers GPU DMA, EOP fence,
PM4 compute dispatch, scalar `s_load` reads, integer ALU, and
`flat_store` writeback against a CPU reference (G15/G22 path).

**PAI-M1 — CLOSED on the proven serial G22 path (without LDS).** This is
the first family of tensor primitives on hardware, not yet the full
Phase 1 tensor runtime (§40). Sub-gates:

| Gate | Primitive | Status |
|------|-----------|--------|
| PAI-M1A | Integer SAXPY (`C[i]=a·A[i]+B[i]`) | VALIDATED (N up to 1M) |
| PAI-M1B | Serial uint32 reduction (`dot_serial`) | VALIDATED (correctness, not throughput) |
| PAI-M1C | Parallel reduction (LDS / wave share) | **BLOCKED** |
| PAI-M1D | Serial-per-row uint32 GEMV | VALIDATED (up to 256×1024) |

M1C unlock condition (do not invent further `COMPUTE_PGM_RSRC2`
values): obtain a real Shader CS AGC blob that allocates LDS and dump
`COMPUTE_PGM_RSRC2` at SH register offset `0x213`. OpenAGC documents
gfx1013 LDS sizing rules but does not ship such a blob. G25 probes
with OpenAGC-minimum 1 KiB (`RSRC2=0x1000C`) still return zero from
LDS on 9.40.

**PAI-M2 / Phase 1 tensor runtime — IN PROGRESS (2026-08-16).**
The float ALU is unlocked (`v_cvt_f32_i32`, `v_add_f32_e64`,
`v_mul_f32_e64`, direct-SGPR e64 form, validated G35/G39/G40/G41)
and the full T4 serial kernel family is hardware-validated 13/13 on
run 004416 (console 9021):

| Family | Kernels | Status |
|--------|---------|--------|
| T4 float (G42-G48) | add1d/sub1d/mul1d/relu/clip/biasadd/matmul | VALIDATED |
| T4 integer (G49-G54) | add2d/sub1d/mul1d/relu/clip/matmul u32 | VALIDATED |

Work breakdown T1-T6 closed (dtype/quant metadata, planner v1, CPU
reference ops, serial GPU kernels, op registry + kernel vtable +
static plan executor, profiler); T7 (MLP 8→16→8 exit test) harness
+ CPU-ref leg green on host, GPU/host-ref differential leg in flight;
T8 (docs) closed.

Open gates before Phase 1 / PAI-M2 work should prioritize:

- MUBUF / flat vector loads (T# still unresolved; flat loads hang);
- LDS unlock via the AGC blob above;
- ~~wave-parallel dispatch beyond the serial `NUM_THREAD_X=1` group
  model~~ — **UNLOCKED (2026-08-17, run 011141/011947, console
  9021)**. First wave-parallel kernels validated: G55 float ramp
  (`c[i]=base+k*i`, NUM_THREAD_X=32, values derived from tid +
  uniform scalars, `v_cvt_f32_i32`+`v_mul/v_add_f32_e64`, commit
  `9f84142`); G56 same kernel fed by `s_load_dword` from a GPU-mem
  header — scalar-read data path inside a 32-thread wave proven, the
  x-side of a wave-parallel GEMV  (`adb11b6`). G57 (per-lane select
  from an `s_load_dwordx16` block via `v_movrels_b32`, the W-side
  unlock for GEMV) in flight, bisected by probe family G58-G63.
  HW esiti (console 9021, run after commit `af22896`): G58
  (blockdump, BLOCK32, 1T, v0..v7) PASS c=10000000+i →
  `s_load_dwordx16` DOES populate s[16:31] with 32 SGPR allocated;
  G59 (vpick, BLOCK32, 32T, v16) FAIL c=0..7 (tid) → direct v16
  read fails even with 32 VGPR allocated; G60/G61 (STANDARD
  RSRC1, 32T, v8/v16) FAIL c=4i+3 (value-path) — CONFOUNDED:
  STANDARD RSRC1 = VGPRS=0 (8 VGPR) and SGPRS=0 (16 SGPR), so
  both the v8/v16 reads and the `s_load_dwordx16` into s[16:31]
  are out of range. **BISECTION CLOSED (G62/G63):** G62 (vpick2,
  BLOCK32, 32T, v8..v15) PASS; G63 (vpick, BLOCK32, 1T, v16) FAIL
  → **hard VGPR ceiling at 16** (v0..v15 usable, v16+ reads tid)
  independent of thread-count and of RSRC1 VGPRS (allocating 32
  VGPRs does not unlock v16+; the field is not a real allocation
  contract on 9.40). Rule for all future kernels: never touch
  v16+. **G64 DECISIVE (commit `fa81eec`):** `v_movrels_b32` with
  the block in-ceiling (v7..v14) and `m0=7` read v[0+7]=v7 for
  ALL lanes — the instruction is **uniform-relative**
  (`v[regno+m0]`), the per-lane index is ignored. DS roundtrip is
  already dead (G25-G27: ds_write/ds_read = 0). **BOTTOM LINE
  9.40 (commits `fa81eec`+`6d1c30c`): no per-lane data-selection
  mechanism exists** — vector loads hang, LDS/DS return 0,
  movrels is uniform-only, VGPR ceiling is 16. Wave-parallel
  kernels can ONLY derive values arithmetically from tid +
  uniform scalars (G55/G56). GEMV stays on the validated G40
  serial-per-row path (1 thread/group, `s_load` per element,
  groups_x=M); the  B-side `decoder_test` (commit `65f29f7`) is
  the differential oracle. Phase 2 plan: RoPE pos tables via the
  G55/G56 ramp (OK), QKV/out GEMM via G40 serial-per-row (cost:
  K*N scalar loads per row — known, not a blocker).
  **G65/G66 UNLOCKED (commit `a8d9089`, run 101812):**
  wave-parallel cos/sin ramp validated. New 9.40 empirical rule:
  `v_cos_f32`/`v_sin_f32` take their operand in TURNS (x2π), not
  radians — the value path multiplies by 2π. Effective GPU
  formula: `c[i] = cos/sin(2π·scale·(4i+3))` (the G15 `(4i+3)`
  lane quirk). HW evidence: `cos(0.15·(4i+3))` expected in
  radians, got cos(162°)=−0.9511 = cos(2π·0.15·3) exact; all 16
  values (8 cos + 8 sin) match <1e-4 with cos²+sin²=1. RoPE
  implication: pos tables ARE generatable on-GPU wave-parallel
  with per-column `scale = inv_freq/(2π)` and the `(4i+3)` lane
  mapping — oracle stays `pai_ref_rope_cossin_f32` (θ=p·inv_freq,
  commit `e157131`). **G67/G68 (commits `d502b10`+`1c45639`):**
  on-GPU RoPE table generator (ropegen.s) — serial-per-element,
  `groups_x=ctx*r2` (128 groups x 1 thread, dispatch 0x8C), header
  (r2, ctx, theta_turns[]) at ud[2:3], C at ud[4:5], cos entry
  writes c[e], sin entry writes the ctx*r2 half.  G67 cos
  **HW-VALIDATED** (run 124611): 128/128 values exact vs the
  `pai_ref_rope_cossin_f32` oracle, EOP ok. G68 sin first failed
  (EOP never fires, holes e=13..15,22..29,32..36, values exact
  where written). **RESOLVED (commit `ca2819e`, run 131807) — NOT
  GPU state, the instruction itself:** G69 patched the `v_sin_f32`
  word (17, 7E026B01) to `v_cos` (7E026D01) in the same kernel →
  PASS bad=0 in the same run; G70 = sin via cos-shift
  (`theta_turns−0.25`, cos(θ−π/2)=sin(θ)) → PASS bad=0, 128/128
  exact. **NEW EMPIRICAL RULE 9.40: `v_sin_f32` is TOXIC in
  serial-per-element (wave hang, dispatch never retires);
  `v_cos_f32` works.** G65/G66 had validated v_sin only in
  wave-parallel (values derived arithmetically, never a serial
  per-element transcendent); the serial ropegen path hangs.
  On-GPU RoPE tables COMPLETE: cos = G67 direct, sin = G70
  cos-shift (same kernel, shifted header). ABI mirror unchanged
  (header theta_turns, sin half at +ctx*r2) — the difference is
  only the sin dispatch header value (tt−0.25); B-side
  `ropegen_diff` section 5 locks the G70 recipe host-side
  (cos mirror + shifted header == oracle sin, max |d| 1.27e-06,
  cos²+sin²=1 preserved).
  **Nonlinear ops contract (commit `7d86a24`, B-side harness
  `nonlinear_contract`):** with no per-lane data select, RMSNorm /
  SiLU / softmax must run as serial per-element kernels (G40/G67
  style), each needing exactly one unprobed primitive: RMSNorm →
  `v_rsqrt_f32` (sum x² via `v_add_f32` + 1/sqrt + mul·gamma),
  SiLU → `v_exp_f32` + reciprocal, softmax → max pass + `v_exp_f32`
  + float sum + division. Float-only margins vs the double oracle:
  RMSNorm max |d| ≤ 2.4e-07 at decoder sizes (5.3e-06 worst-case
  |x|~100, n=2048) — tolerance 1e-4 safe with float accumulation;
  SiLU bit-identical; softmax row sum 1.000000015, finite at scores
  ±30; causal attention rows finite, out within v-range. This is the
  target contract for the `v_rsqrt_f32`/`v_exp_f32` probes.
  Empirical note: RSRC1 `VGPRS` must
  cover the VGPRs a kernel actually touches (G55/G56 use v1-v6,
  lanepick v16-v31 → `PAI_EXP_RSRC1_VGPR32=0x602C0003`).

Measured serial GEMV submit→EOP bandwidth is apparent (~2–4 GB/s), not
HBM peak; treat it as a harness timing signal only.

---

## 44. Project Definition

**Name:** ProsperoAI  
**Native model extension:** `.pai`  
**Console executable:** `prosperoai.elf`  
**Core language:** C  
**Desktop:** Electron + TypeScript + Rust  
**Primary compute:** PS5 GPU  
**Secondary compute:** Zen 2 CPU  
**Primary workload:** LLM inference  
**Architectural scope:** General-purpose AI runtime  
**Deployment:** Single modular PS5 payload  
**Development model:** Private / unlicensed during initial development  
**Primary objective:** Maximum achievable AI inference performance on jailbroken PS5 hardware