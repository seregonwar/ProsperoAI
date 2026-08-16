/*
 * ProsperoAI — Scheduler (whitepaper §18) and execution planning (§11)
 *
 * The scheduler turns a compiled graph + its static memory plan into an
 * Execution Plan: an ordered list of op executions with CPU/GPU device
 * placement and per-step memory accounting. It exposes the §18 policies
 * (Interactive / Throughput / Exclusive / Balanced), the §17 memory
 * modes (Performance / Balanced / Capacity), and a minimal session
 * registry with priority ordering (continuous batching seed, Phase 8).
 *
 * The scheduler also ships a reference executor: it runs an Execution
 * Plan against the CPU reference backend over the memory-plan region.
 * This closes the Phase 1 loop — "run a small synthetic neural network
 * completely through ProsperoAI" — on host builds. GPU dispatch and
 * kernel-level execution arrive with the GPU backend (Phase 1+); the
 * `device` field of every step is a placement *decision* that the
 * runtime then honors.
 */

#ifndef PAI_SCHEDULER_H
#define PAI_SCHEDULER_H

#include <pai/error.h>

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
 * Build the execution plan for a graph + its memory plan.
 *
 * Steps are emitted in topological order (epoch 0..n-1); each step
 * carries the device placement decision:
 *   - an explicit per-op override from `device_hints` (indexed by op
 *     id, pai_sched_device_t; NULL = none) always wins;
 *   - otherwise the per-op-kind default (GEMM/GEMV/MATMUL/ATTENTION/
 *     SOFTMAX -> GPU, the rest -> CPU);
 *   - PAI_SCHED_EXCLUSIVE additionally forces compute ops onto the GPU
 *     unless a hint overrides them.
 *
 * `mem_plan` supplies region/naive/peak statistics and (for the
 * executor) the per-value offsets. Runs the topo sort first if needed.
 */
pai_status_t pai_sched_build(pai_graph_t *graph,
                             const pai_graph_mem_plan_t *mem_plan,
                             pai_sched_policy_t policy,
                             const uint8_t *device_hints,
                             pai_sched_plan_t *out);

/*
 * Persistent variant for session execution: per-step live/produced/
 * released byte accounting uses the persistent lifetime view (inputs
 * and params live until execution end), matching the layout a session
 * runs repeatedly against (pai_graph_memory_plan_persistent). Use this
 * when the executor will re-run the plan (e.g. one generate step per
 * call) so peak_live and the memory-mode check agree with the region.
 */
pai_status_t pai_sched_build_persistent(pai_graph_t *graph,
                                        const pai_graph_mem_plan_t *mem_plan,
                                        pai_sched_policy_t policy,
                                        const uint8_t *device_hints,
                                        pai_sched_plan_t *out);

/*
 * Memory mode for a residency budget (§17): PERFORMANCE when the whole
 * planned region fits, BALANCED when at least the peak-live working set
 * fits, CAPACITY otherwise.
 */
pai_sched_memory_mode_t pai_sched_plan_memory_mode(const pai_sched_plan_t *plan,
                                                   uint64_t budget);

/*
 * Can this plan execute within `budget` bytes of storage? Requires at
 * least the peak-live working set.
 */
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
 * Execute the plan against the CPU reference backend. `region` is a
 * buffer of at least plan->region_bytes bytes; every value lives at
 * region + pai_graph_mem_plan_offset(mem_plan, value_id).
 *
 * Supported ops: ADD, MUL, GEMM, MATMUL, GEMV, RELU, SOFTMAX, RMSNORM,
 * LAYERNORM, CONCAT, COPY, RESHAPE, CONVERT (same-dtype). ROPE,
 * ATTENTION and CUSTOM return PAI_ERR_UNSUPPORTED in v0.
 */
pai_status_t pai_sched_execute(const pai_graph_t *graph,
                               const pai_graph_mem_plan_t *mem_plan,
                               const pai_sched_plan_t *plan, uint8_t *region,
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

/*
 * Pick the highest-priority active session (ties broken by lowest id).
 * Returns the slot index, or -1 when no session is active.
 */
int pai_sched_pick_session(const pai_sched_sessions_t *sessions);

#ifdef __cplusplus
}
#endif

#endif /* PAI_SCHEDULER_H */
