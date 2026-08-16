/*
 * ProsperoAI — plan profiler (T6, whitepaper §30/§31).
 *
 * Per-step execution profile of a static plan: CPU/GPU wall times,
 * transfer counters, a human-readable plan profile log and a
 * deterministic replay hash over the plan structure (op ids, devices,
 * transfer payloads — deliberately excluding wall times, so the same
 * plan/profile replays to the same hash).
 *
 * On failure mid-run the profile is left unfinished (hash 0, partial
 * totals): callers must discard it on error.
 *
 * Timing comes from pai_sched_execute_hooked (scheduler) or the
 * caller's own measurements; this module owns the accounting.
 */

#ifndef PAI_PROFILER_PROFILE_H
#define PAI_PROFILER_PROFILE_H

#include <pai/error.h>

#include <scheduler/scheduler.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PAI_PROFILE_MAX_STEPS PAI_SCHED_MAX_STEPS
#define PAI_PROFILE_LOG_MAX   4096u

/* One recorded step. */
typedef struct pai_profile_step {
  uint32_t op_id;
  uint8_t  device;          /* pai_sched_device_t                        */
  uint64_t cpu_wall_ns;     /* measured CPU-side wall (0 for GPU steps)  */
  uint64_t gpu_wall_ns;     /* measured GPU-side wall (0 on host)        */
  uint64_t transfer_bytes;  /* payload moved for the step                */
} pai_profile_step_t;

typedef struct pai_profile {
  uint32_t          num_steps;
  pai_profile_step_t steps[PAI_PROFILE_MAX_STEPS];
  uint64_t          total_cpu_ns;
  uint64_t          total_gpu_ns;
  uint64_t          total_transfer_bytes;
  uint64_t          total_ns;        /* wall of the whole run           */
  uint64_t          hash;            /* FNV-1a replay hash (0 when empty) */
  char              log[PAI_PROFILE_LOG_MAX]; /* plan profile log       */
} pai_profile_t;

void pai_profile_init(pai_profile_t *profile);

/*
 * Record one executed step. cpu_ns/gpu_ns are the wall times measured
 * on each device (pass 0 for the device not involved); transfer_bytes
 * is the payload moved for the step (0 for host-resident ops). Appends
 * a log line. PAI_ERR_NOMEM when the step table is full.
 */
pai_status_t pai_profile_add_step(pai_profile_t *profile, uint32_t op_id,
                                  uint8_t device, uint64_t cpu_ns,
                                  uint64_t gpu_ns, uint64_t transfer_bytes);

/*
 * Finalize the profile: totals + replay hash over the structural
 * records (op_id, device, transfer_bytes per step). Returns the log
 * length (0 when empty).
 */
uint32_t pai_profile_finish(pai_profile_t *profile);

/*
 * Run the plan through the reference executor measuring per-step CPU
 * wall time (via pai_sched_execute_hooked) and building the profile.
 * `transfer_bytes_per_step` is credited to GPU-placed steps (0 = none).
 * On success the profile is finalized (finish applied).
 */
pai_status_t pai_profile_run_plan(const pai_graph_t *graph,
                                  const pai_graph_mem_plan_t *mem_plan,
                                  const pai_sched_plan_t *plan,
                                  uint8_t *region,
                                  uint64_t transfer_bytes_per_step,
                                  pai_profile_t *out,
                                  pai_sched_run_stats_t *out_stats);

#ifdef __cplusplus
}
#endif

#endif /* PAI_PROFILER_PROFILE_H */
