/*
 * ProsperoAI — plan profiler (T6).
 */

#include "profile.h"

#include <stdio.h>
#include <string.h>

/* FNV-1a 64 (deterministic, cheap; fine for a replay identity). */
static uint64_t
fnv1a64_update(uint64_t hash, const void *data, size_t nbytes) {
  const uint8_t *p = (const uint8_t *)data;
  for (size_t i = 0; i < nbytes; i++) {
    hash ^= p[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

void
pai_profile_init(pai_profile_t *profile) {
  if (profile == NULL) {
    return;
  }
  memset(profile, 0, sizeof(*profile));
  profile->log[0] = '\0';
}

pai_status_t
pai_profile_add_step(pai_profile_t *profile, uint32_t op_id, uint8_t device,
                     uint64_t cpu_ns, uint64_t gpu_ns,
                     uint64_t transfer_bytes) {
  pai_profile_step_t *step;
  int written;

  if (profile == NULL || op_id == 0) {
    return PAI_ERR_INVALID_ARG;
  }
  if (profile->num_steps >= PAI_PROFILE_MAX_STEPS) {
    return PAI_ERR_NOMEM;
  }

  step = &profile->steps[profile->num_steps];
  step->op_id = op_id;
  step->device = device;
  step->cpu_wall_ns = cpu_ns;
  step->gpu_wall_ns = gpu_ns;
  step->transfer_bytes = transfer_bytes;
  profile->num_steps++;

  profile->total_cpu_ns += cpu_ns;
  profile->total_gpu_ns += gpu_ns;
  profile->total_transfer_bytes += transfer_bytes;

  written = snprintf(profile->log + strlen(profile->log),
                     sizeof(profile->log) - strlen(profile->log),
                     "step %u op=%u dev=%s cpu=%llu gpu=%llu xfer=%llu\n",
                     profile->num_steps, op_id,
                     pai_sched_device_name(device),
                     (unsigned long long)cpu_ns, (unsigned long long)gpu_ns,
                     (unsigned long long)transfer_bytes);
  if (written < 0 ||
      (size_t)written >= sizeof(profile->log) - strlen(profile->log)) {
    /* Log full: truncate cleanly and keep accounting. */
    profile->log[sizeof(profile->log) - 1] = '\0';
  }
  return PAI_OK;
}

uint32_t
pai_profile_finish(pai_profile_t *profile) {
  uint64_t hash = UINT64_C(1469598103934665603); /* FNV offset basis */
  if (profile == NULL) {
    return 0;
  }
  profile->total_ns = profile->total_cpu_ns + profile->total_gpu_ns;

  /* Replay hash over the structural records only (no wall times). */
  for (uint32_t i = 0; i < profile->num_steps; i++) {
    const pai_profile_step_t *step = &profile->steps[i];
    uint32_t op_id = step->op_id;
    uint8_t device = step->device;
    uint64_t xfer = step->transfer_bytes;
    hash = fnv1a64_update(hash, &op_id, sizeof(op_id));
    hash = fnv1a64_update(hash, &device, sizeof(device));
    hash = fnv1a64_update(hash, &xfer, sizeof(xfer));
  }
  profile->hash = hash;
  return (uint32_t)strlen(profile->log);
}

/* Hook ctx: profile being built. */
typedef struct pai_profile_hook_ctx {
  pai_profile_t *profile;
  uint64_t transfer_per_step;
  uint32_t steps; /* sanity guard against hook/executor drift */
} pai_profile_hook_ctx_t;

static void
profile_step_hook(void *ctx, uint32_t step_index, pai_graph_op_id op_id,
                  uint8_t device, uint64_t wall_ns) {
  pai_profile_hook_ctx_t *pc = (pai_profile_hook_ctx_t *)ctx;
  uint64_t xfer = 0;
  (void)step_index;
  if (device == PAI_SCHED_DEV_GPU) {
    xfer = pc->transfer_per_step;
  }
  if (pc->profile != NULL &&
      pai_profile_add_step(pc->profile, op_id, device, wall_ns, 0, xfer) ==
          PAI_OK) {
    pc->steps++;
  }
}

pai_status_t
pai_profile_run_plan(const pai_graph_t *graph,
                     const pai_graph_mem_plan_t *mem_plan,
                     const pai_sched_plan_t *plan, uint8_t *region,
                     uint64_t transfer_bytes_per_step, pai_profile_t *out,
                     pai_sched_run_stats_t *out_stats) {
  pai_profile_hook_ctx_t pc;
  pai_status_t st;

  if (graph == NULL || mem_plan == NULL || plan == NULL || region == NULL ||
      out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (plan->num_steps != graph->num_ops) {
    return PAI_ERR_MISMATCH;
  }

  pai_profile_init(out);
  pc.profile = out;
  pc.transfer_per_step = transfer_bytes_per_step;
  pc.steps = 0;

  st = pai_sched_execute_hooked(graph, mem_plan, plan, region, out_stats,
                                profile_step_hook, &pc);
  if (st != PAI_OK) {
    return st;
  }
  if (pc.steps != plan->num_steps) {
    return PAI_ERR_INTERNAL; /* hook never fires for every step */
  }
  pai_profile_finish(out);
  return PAI_OK;
}
