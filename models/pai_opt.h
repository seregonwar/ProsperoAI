/*
 * ProsperoAI — mixed-precision quantization planner (whitepaper §15/§20).
 *
 * `pai optimize` reads a `.pai` container's manifest and computes, per
 * weight tensor, the on-disk size under every v0 storage scheme (f32,
 * q8, q4 with a configurable scale-group size). Tensors are classified
 * by role (embedding / attention / mlp / output / norm / other) so the
 * planner can apply the §15 policy — keep sensitive layers exact or
 * lightly quantized, spend the aggressive quantizations where they
 * save the most bytes. Two modes:
 *
 *   - default: the §15 mixed-precision plan (embedding q8, attention
 *     q8, mlp q4, output q8, norms f32), never re-quantizing a tensor
 *     that is already stored more cheaply;
 *   - budget: greedy largest-savings-first selection from the current
 *     storage until the total fits the budget (or the cheapest
 *     feasible layout when the budget is unreachable).
 *
 * The plan is deterministic and side-effect free: it never rewrites
 * the container. Desktop consumes the JSON report and re-runs
 * `pai convert` with the chosen quantization per weight group.
 */

#ifndef PAI_MODELS_PAI_OPT_H
#define PAI_MODELS_PAI_OPT_H

#include <ir/ir.h>
#include <pai/error.h>
#include <pai/pai.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PAI_OPT_MAX_TENSORS PAI_PAI_MAX_TENSORS

typedef enum pai_opt_role {
  PAI_OPT_ROLE_EMBEDDING = 0,
  PAI_OPT_ROLE_ATTENTION,
  PAI_OPT_ROLE_MLP,
  PAI_OPT_ROLE_OUTPUT,
  PAI_OPT_ROLE_NORM,
  PAI_OPT_ROLE_OTHER,
  PAI_OPT_ROLE_COUNT
} pai_opt_role_t;

typedef enum pai_opt_scheme {
  PAI_OPT_SCHEME_F32 = 0, /* 32-bit float canonical                 */
  PAI_OPT_SCHEME_Q8,      /* 8-bit symmetric per-group (§15)        */
  PAI_OPT_SCHEME_Q4,      /* 4-bit packed, symmetric per-group      */
  PAI_OPT_SCHEME_COUNT
} pai_opt_scheme_t;

typedef struct pai_opt_tensor {
  char name[64];
  uint32_t value_id;
  pai_opt_role_t role;
  uint64_t numel;
  uint64_t cur_bytes;
  pai_opt_scheme_t cur_scheme;       /* best known from storage/IR    */
  uint64_t size_bytes[PAI_OPT_SCHEME_COUNT];
  pai_opt_scheme_t chosen;
} pai_opt_tensor_t;

typedef struct pai_opt_plan {
  char name[PAI_PAI_NAME_MAX + 1];
  uint32_t num_tensors;
  pai_opt_tensor_t tensors[PAI_OPT_MAX_TENSORS];
  uint64_t budget;                   /* 0 = none (role defaults)      */
  uint64_t cur_total;
  uint64_t plan_total;
  uint64_t min_total;                /* cheapest feasible layout      */
  int feasible;                      /* plan_total <= budget (budget) */
} pai_opt_plan_t;

/* Stable role name for reports. */
const char *pai_opt_role_name(pai_opt_role_t role);

/* Stable scheme name for reports. */
const char *pai_opt_scheme_name(pai_opt_scheme_t scheme);

/* Classify a manifest tensor name into a role (§15 table). */
pai_opt_role_t pai_opt_classify(const char *tensor_name);

/*
 * Build the plan for an opened container. `ir` may be NULL (the
 * current scheme is then inferred from storage sizes); when present,
 * per-value quantization metadata refines it. `budget` 0 selects the
 * role-default plan, otherwise a greedy largest-savings fit.
 * `group_size` 0 = per-tensor scale block (PAI_QUANT_GROUP_AUTO).
 */
pai_status_t pai_opt_plan(const pai_pai_container_t *c,
                          const pai_ir_program_t *ir, uint64_t budget,
                          uint16_t group_size, pai_opt_plan_t *out);

#ifdef __cplusplus
}
#endif

#endif /* PAI_MODELS_PAI_OPT_H */
