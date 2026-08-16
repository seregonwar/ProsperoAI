/*
 * ProsperoAI — Scheduler (whitepaper §18) and execution planning (§11)
 *
 * Turns a compiled graph + its static memory plan into an Execution
 * Plan: ordered steps with CPU/GPU placement and per-step memory
 * accounting, using the §18 policies and §17 memory modes, plus a
 * minimal priority session registry (batching seed, Phase 8). Ships a
 * reference executor that runs a plan against the CPU reference
 * backend; the per-step `device` is a placement decision the runtime
 * honors (GPU dispatch arrives with the GPU backend).
 */

#ifndef PAI_SCHEDULER_H
#define PAI_SCHEDULER_H

#include <pai/error.h>
#include <pai/op.h>

#include <graph/graph.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Policies (§18) and devices                                          */
/* ------------------------------------------------------------------ */

typedef enum pai_sched_policy {
  PAI_SCHED_INTERACTIVE = 0, /* prioritize low response latency          */
  PAI_SCHED_THROUGHPUT,      /* prioritize aggregate tokens/sec          */
  PAI_SCHED_EXCLUSIVE,       /* dedicate resources to a single workload  */
  PAI_SCHED_BALANCED,        /* trade latency, concurrency and memory    */
} pai_sched_policy_t;

typedef enum pai_sched_device {
  PAI_SCHED_DEV_ANY = 0, /* no preference; default placement applies     */
  PAI_SCHED_DEV_CPU,
  PAI_SCHED_DEV_GPU,
} pai_sched_device_t;

/* Memory modes (§17). Derived from a residency budget; see
 * pai_sched_plan_memory_mode. */
typedef enum pai_sched_memory_mode {
  PAI_SCHED_MEM_PERFORMANCE = 0, /* whole planned region resident        */
  PAI_SCHED_MEM_BALANCED,        /* peak-live set resident, partial      */
  PAI_SCHED_MEM_CAPACITY,        /* must spill: hierarchical VM (§17)    */
} pai_sched_memory_mode_t;

/* ------------------------------------------------------------------ */
/* Execution plan                                                      */
/* ------------------------------------------------------------------ */

#define PAI_SCHED_MAX_STEPS PAI_GRAPH_MAX_OPS

typedef struct pai_sched_step {
  uint32_t op_id;         /* graph op id                                 */
  uint8_t  device;        /* pai_sched_device_t placement decision       */
  uint32_t epoch;         /* topological position                        */
  uint64_t live_bytes;    /* storage live at this step (incl. produced)  */
  uint64_t produced_bytes;/* outputs of this step                         */
  uint64_t released_bytes;/* storage freed when this step begins          */
} pai_sched_step_t;

typedef struct pai_sched_plan {
  uint32_t          num_steps;
  pai_sched_step_t  steps[PAI_SCHED_MAX_STEPS];
  pai_sched_policy_t policy;
  uint64_t          region_bytes;  /* planned storage region (reused)    */
  uint64_t          naive_bytes;   /* sum of all value sizes (no reuse)  */
  uint64_t          peak_live;     /* max concurrent storage             */
  uint32_t          gpu_steps;     /* steps placed on GPU                */
  uint32_t          cpu_steps;     /* steps placed on CPU                */
} pai_sched_plan_t;

const char *pai_sched_policy_name(pai_sched_policy_t policy);
const char *pai_sched_device_name(uint8_t device);
const char *pai_sched_memory_mode_name(uint8_t mode);

/*
 * Build the execution plan. Steps are emitted in topological order;
 * each carries a device decision: an explicit per-op entry in
 * `device_hints` (indexed by op id, pai_sched_device_t, NULL = none)
 * wins, otherwise the per-kind default (GEMM/GEMV/MATMUL/ATTENTION/
 * SOFTMAX -> GPU, rest -> CPU); PAI_SCHED_EXCLUSIVE additionally
 * forces compute ops onto the GPU unless hinted. `mem_plan` supplies
 * region/naive/peak stats and the executor's per-value offsets. Runs
 * the topo sort first if needed.
 */
pai_status_t pai_sched_build(pai_graph_t *graph,
                             const pai_graph_mem_plan_t *mem_plan,
                             pai_sched_policy_t policy,
                             const uint8_t *device_hints,
                             pai_sched_plan_t *out);

/* Persistent variant: per-step live/produced/released accounting uses
 * the persistent lifetime view (inputs and params live until execution
 * end), matching the layout a session re-runs. Use when the executor
 * will re-run the plan so peak_live agrees with the region. */
pai_status_t pai_sched_build_persistent(pai_graph_t *graph,
                                        const pai_graph_mem_plan_t *mem_plan,
                                        pai_sched_policy_t policy,
                                        const uint8_t *device_hints,
                                        pai_sched_plan_t *out);

/* Memory mode for a residency budget (§17): PERFORMANCE when the whole
 * planned region fits, BALANCED when the peak-live set fits, CAPACITY
 * otherwise. */
pai_sched_memory_mode_t pai_sched_plan_memory_mode(const pai_sched_plan_t *plan,
                                                   uint64_t budget);

/* Can this plan run within `budget` bytes (at least peak-live)? */
pai_status_t pai_sched_plan_check(const pai_sched_plan_t *plan,
                                  uint64_t budget);

/* ------------------------------------------------------------------ */
/* Reference executor (Phase 1 vertical slice)                         */
/* ------------------------------------------------------------------ */

typedef struct pai_sched_run_stats {
  uint64_t steps_executed; /* steps actually run                         */
  uint64_t gpu_steps;      /* placement: GPU (not dispatched here)       */
  uint64_t cpu_steps;      /* placement: CPU                             */
} pai_sched_run_stats_t;

/*
 * Execute the plan against the CPU reference backend. `region` holds
 * at least plan->region_bytes; values live at region +
 * pai_graph_mem_plan_offset(mem_plan, value_id). Supported: ADD, MUL,
 * GEMM, MATMUL, GEMV, RELU, SOFTMAX, RMSNORM, LAYERNORM, CONCAT, COPY,
 * RESHAPE, CONVERT (same-dtype); ROPE, ATTENTION and CUSTOM are
 * unsupported in v0.
 */
pai_status_t pai_sched_execute(const pai_graph_t *graph,
                               const pai_graph_mem_plan_t *mem_plan,
                               const pai_sched_plan_t *plan, uint8_t *region,
                               pai_sched_run_stats_t *out_stats);

/* Per-step hook invoked after each successfully executed step with its
 * measured wall time (monotonic, ns). NULL disables; used by the T6
 * profiler to build per-step profiles without re-running the plan. */
typedef void (*pai_sched_step_fn)(void *ctx, uint32_t step_index,
                                  pai_graph_op_id op_id, uint8_t device,
                                  uint64_t wall_ns);

/* pai_sched_execute with an optional per-step hook (profiling). */
pai_status_t pai_sched_execute_hooked(const pai_graph_t *graph,
                                      const pai_graph_mem_plan_t *mem_plan,
                                      const pai_sched_plan_t *plan,
                                      uint8_t *region,
                                      pai_sched_run_stats_t *out_stats,
                                      pai_sched_step_fn step_fn, void *ctx);

/* ------------------------------------------------------------------ */
/* Kernel vtable (T5B)                                                */
/* ------------------------------------------------------------------ */

/*
 * One resolved kernel slot: the registry entry selected for a graph op
 * (T5A registry x T5B selection). `device` is the default placement
 * (same policy as pai_sched_build without hints).
 */
typedef struct pai_kernel_entry {
  pai_graph_op_kind_t kind;   /* graph op kind                          */
  const char        *op_name; /* registry name, e.g. "vecadd_f32"       */
  pai_kernel_id_t    kernel_id; /* PAI_KERNEL_* or PAI_KERNEL_NONE      */
  uint8_t            device;  /* pai_sched_device_t default placement   */
  uint32_t           caps;    /* PAI_OP_CAP_* advertised by the op      */
} pai_kernel_entry_t;

/*
 * Select the kernel for a graph op through the op registry. `reg` may
 * be NULL to use the builtin Phase-1 registry (built on the stack).
 * Unregistered kinds (PAI_OP_CUSTOM) resolve to PAI_ERR_UNSUPPORTED.
 */
pai_status_t pai_kernel_select(const pai_graph_t *graph,
                               pai_graph_op_id op_id,
                               const pai_op_registry_t *reg,
                               pai_kernel_entry_t *out);

/*
 * Capability gate over a whole plan: every step's op must resolve in
 * the registry and advertise every flag in `caps_required`. NULL `reg`
 * uses the builtin registry. Catches ops with no implementation for
 * the target backend before any memory is touched.
 */
pai_status_t pai_kernel_plan_check(const pai_graph_t *graph,
                                   const pai_sched_plan_t *plan,
                                   uint32_t caps_required,
                                   const pai_op_registry_t *reg);

/*
 * Vtable-gated reference executor: runs pai_kernel_plan_check with
 * PAI_OP_CAP_CPU_REF, then pai_sched_execute. Same result as
 * pai_sched_execute for supported graphs, PAI_ERR_UNSUPPORTED when any
 * op lacks a CPU reference implementation.
 */
pai_status_t pai_sched_execute_vtable(const pai_graph_t *graph,
                                      const pai_graph_mem_plan_t *mem_plan,
                                      const pai_sched_plan_t *plan,
                                      uint8_t *region,
                                      pai_sched_run_stats_t *out_stats);

/* ------------------------------------------------------------------ */
/* Session registry (§18: session priority, batching seed)             */
/* ------------------------------------------------------------------ */

#define PAI_SCHED_MAX_SESSIONS 16u

typedef struct pai_sched_session {
  uint64_t id;
  int32_t  priority;  /* higher = more important                        */
  uint8_t  policy;    /* pai_sched_policy_t                             */
  uint8_t  active;    /* workload in flight                             */
} pai_sched_session_t;

typedef struct pai_sched_sessions {
  pai_sched_session_t slots[PAI_SCHED_MAX_SESSIONS];
  uint32_t count;
  uint64_t next_id;
} pai_sched_sessions_t;

void pai_sched_sessions_init(pai_sched_sessions_t *sessions);

pai_status_t pai_sched_session_create(pai_sched_sessions_t *sessions,
                                      int32_t priority, uint8_t policy,
                                      uint64_t *out_id);
pai_status_t pai_sched_session_destroy(pai_sched_sessions_t *sessions,
                                       uint64_t id);
int  pai_sched_session_active(const pai_sched_sessions_t *sessions,
                              uint64_t id);

/* Highest-priority active session (ties: lowest id); slot index or -1. */
int pai_sched_pick_session(const pai_sched_sessions_t *sessions);

#ifdef __cplusplus
}
#endif

#endif /* PAI_SCHEDULER_H */
