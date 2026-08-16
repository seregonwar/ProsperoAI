/*
 * ProsperoAI — `pai` command-line tool (whitepaper §20 official tooling)
 *
 *   pai convert  <desc.txt> <out.pai> [--quant q8|q4] [--group N] [--compress kraken|none]
 *   pai inspect  <model.pai> [--json]
 *   pai validate <model.pai>
 *   pai optimize <model.pai> [--budget B] [--group N] [--json]
 *   pai benchmark <model.pai> [--steps N] [--prompt P]
 *
 * The host-side tooling behind ProsperoAI Desktop (§7): model import
 * (conversion + quantization + .pai packaging), container inspection,
 * mixed-precision planning (§15/§20), full validation, and
 * reproducible generation benchmarking (§31).
 */

#include <bench.h>
#include <compression/pai_compress.h>
#include <gateway/gateway.h>
#include <importer.h>
#include <model.h>

#include <llama.h>

#include <gateway/json.h>
#include <ir/ir.h>
#include <pai/api.h>
#include <pai/dtype.h>
#include <pai/error.h>
#include <pai/pai.h>
#include <pai/version.h>
#include <pai_opt.h>
#include <protocol/protocol.h>
#include <scheduler/scheduler.h>

#include <tokenizer.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
usage(void) {
  printf(
      "pai — ProsperoAI model tooling\n"
      "\n"
      "usage:\n"
      "  pai convert  <desc.txt> <out.pai> [--quant q8|q4] [--group N] [--compress kraken|none]\n"
      "               import a model description + raw weights into a .pai\n"
      "  pai inspect  <model.pai> [--json]  dump container contents\n"
      "  pai validate <model.pai>            full integrity + load check\n"
      "  pai optimize <model.pai> [--budget B] [--group N] [--json]\n"
      "               mixed-precision plan (§15): per-tensor f32/q8/q4\n"
      "               sizes; --budget fits B bytes (K/M/G/T or NN% of\n"
      "               current total), greedy largest-savings first\n"
      "  pai benchmark <model.pai> [--steps N] [--prompt P]\n"
      "               time end-to-end generation\n"
      "  pai serve    <model.pai>... [--remote [name@]host:port]...\n"
      "               [--host H] [--port N]  OpenAI-compatible gateway (§26)\n"
      "  pai proto-ping <host> <port> [--count N]\n"
      "               Prospero Protocol connectivity check (§24/§25)\n");
}

static int cmd_inspect_json(const char *path); /* defined below       */

/* ------------------------------------------------------------------ */
/* convert                                                             */
/* ------------------------------------------------------------------ */

static int
cmd_convert(int argc, char **argv) {
  const char *desc;
  const char *out;
  pai_import_options_t opts;
  pai_status_t st;
  int i;

  if (argc < 4) {
    usage();
    return 2;
  }
  desc = argv[2];
  out = argv[3];
  memset(&opts, 0, sizeof(opts));

  for (i = 4; i < argc; i++) {
    if (strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
      opts.quant = argv[++i];
    } else if (strcmp(argv[i], "--group") == 0 && i + 1 < argc) {
      opts.quant_group = (uint16_t)strtoul(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "--compress") == 0 && i + 1 < argc) {
      if (strcmp(argv[i + 1], "kraken") == 0) {
        opts.compress = PAI_COMP_METHOD_KRAKEN;
      } else if (strcmp(argv[i + 1], "none") == 0) {
        opts.compress = 0;
      } else {
        usage();
        return 2;
      }
      i++;
    } else {
      usage();
      return 2;
    }
  }

  /* Route by magic: GGUF -> llama.cpp adapter; otherwise the text
   * model-description DSL (whitepaper §8/§9.3). */
  {
    FILE *probe = fopen(desc, "rb");
    uint8_t head[4] = {0, 0, 0, 0};
    int is_gguf = 0;
    if (probe != NULL) {
      is_gguf = fread(head, 1, 4, probe) == 4 && pai_llama_is_gguf(head);
      fclose(probe);
    }
    if (is_gguf) {
      pai_container_quant_t q;
      q.quant = opts.quant;
      q.quant_group = opts.quant_group;
      q.compress = opts.compress;
      st = pai_llama_import(desc, out, &q);
    } else {
      st = pai_import_model_to_file(desc, out, &opts);
    }
  }
  if (st != PAI_OK) {
    fprintf(stderr, "convert: %s\n", pai_status_str(st));
    return 1;
  }
  printf("converted %s -> %s%s%s%s\n", desc, out,
         opts.quant != NULL ? " (quant=" : "",
         opts.quant != NULL ? opts.quant : "",
         opts.quant != NULL ? ")" : "");
  return 0;
}

/* ------------------------------------------------------------------ */
/* inspect                                                             */
/* ------------------------------------------------------------------ */

static void
print_shape(const uint64_t *shape, uint32_t rank) {
  printf("[");
  for (uint32_t i = 0; i < rank; i++) {
    if (i) {
      printf(",");
    }
    printf("%llu", (unsigned long long)shape[i]);
  }
  printf("]");
}

static int
cmd_inspect(int argc, char **argv) {
  const char *path;
  uint8_t *blob = NULL;
  uint32_t nbytes = 0;
  pai_pai_container_t c;
  pai_pai_meta_t meta;
  pai_pai_tensor_t tensors[PAI_PAI_MAX_TENSORS];
  uint32_t num_tensors = 0;
  pai_ir_program_t ir;
  pai_tok_t tok;
  const uint8_t *sec;
  uint32_t sec_size;
  pai_status_t st;

  if (argc < 3) {
    usage();
    return 2;
  }
  path = argv[2];

  if (argc >= 4 && strcmp(argv[3], "--json") == 0) {
    return cmd_inspect_json(path);
  }

  st = pai_pai_read_file(path, &blob, &nbytes);
  if (st != PAI_OK) {
    fprintf(stderr, "inspect: %s\n", pai_status_str(st));
    return 1;
  }
  st = pai_pai_open(blob, nbytes, &c);
  if (st != PAI_OK) {
    fprintf(stderr, "inspect: %s\n", pai_status_str(st));
    free(blob);
    return 1;
  }

  printf("container: %s (%u bytes, version %u)\n", path, c.nbytes, c.version);
  for (uint32_t i = 0; i < c.num_sections; i++) {
    printf("  section %-10s offset %-8u size %u\n",
           pai_pai_section_name(c.sections[i].type), c.sections[i].offset,
           c.sections[i].size);
  }

  sec = pai_pai_section(&c, PAI_PAI_SEC_META, &sec_size);
  if (sec != NULL && pai_pai_meta_decode(sec, sec_size, &meta) == PAI_OK) {
    printf("meta: name=%s family=%u context=%u layers=%u kv_bytes=%u "
           "vocab=%u\n",
           meta.name, meta.family, meta.context_len, meta.num_layers,
           meta.kv_bytes_per_token, meta.vocab_size);
  }

  sec = pai_pai_section(&c, PAI_PAI_SEC_MANIFEST, &sec_size);
  if (sec != NULL &&
      pai_pai_manifest_decode(sec, sec_size, tensors, PAI_PAI_MAX_TENSORS,
                              &num_tensors) == PAI_OK) {
    printf("manifest: %u tensors\n", num_tensors);
    for (uint32_t i = 0; i < num_tensors; i++) {
      const pai_pai_tensor_t *t = &tensors[i];
      printf("  %-24s value=%u %s ", t->name, t->value_id,
             pai_dtype_name(t->dtype));
      print_shape(t->shape, t->rank);
      printf(" offset=%llu size=%llu\n", (unsigned long long)t->offset,
             (unsigned long long)t->size_bytes);
    }
  }

  sec = pai_pai_section(&c, PAI_PAI_SEC_IR, &sec_size);
  if (sec != NULL && pai_ir_decode(sec, sec_size, &ir) == PAI_OK) {
    printf("ir: %u values, %u ops, %u inputs, %u outputs\n", ir.num_values,
           ir.num_ops, ir.num_inputs, ir.num_outputs);
    for (uint32_t i = 1; i <= ir.num_ops; i++) {
      const pai_ir_op_t *o = &ir.ops[i];
      printf("  op %-3u %-10s %u inputs %u outputs\n", i,
             pai_ir_op_kind_name(o->kind), o->num_inputs, o->num_outputs);
    }
  }

  pai_tok_init(&tok);
  sec = pai_pai_section(&c, PAI_PAI_SEC_TOKENIZER, &sec_size);
  if (sec != NULL && pai_tok_deserialize(&tok, sec, sec_size) == PAI_OK) {
    printf("tokenizer: %u tokens, %u merges\n", tok.num_tokens, tok.num_merges);
  }
  pai_tok_destroy(&tok);

  free(blob);
  return 0;
}

/* ------------------------------------------------------------------ */
/* inspect --json (machine-readable report for Desktop §7 tooling)     */
/* ------------------------------------------------------------------ */

static void
json_shape(pai_json_wb_t *wb, const uint64_t *shape, uint32_t rank) {
  uint32_t i;
  pai_json_wb_puts(wb, "[");
  for (i = 0; i < rank; i++) {
    if (i) {
      pai_json_wb_puts(wb, ",");
    }
    pai_json_wb_putf(wb, "%llu", (unsigned long long)shape[i]);
  }
  pai_json_wb_puts(wb, "]");
}

static int
cmd_inspect_json(const char *path) {
  uint8_t *blob = NULL;
  uint32_t nbytes = 0;
  pai_pai_container_t c;
  pai_pai_meta_t meta;
  pai_pai_tensor_t tensors[PAI_PAI_MAX_TENSORS];
  uint32_t num_tensors = 0;
  pai_ir_program_t ir;
  pai_tok_t tok;
  const uint8_t *sec;
  uint32_t sec_size;
  pai_json_wb_t wb;
  pai_status_t st;
  uint32_t i;

  st = pai_pai_read_file(path, &blob, &nbytes);
  if (st != PAI_OK) {
    fprintf(stderr, "inspect: %s\n", pai_status_str(st));
    return 1;
  }
  st = pai_pai_open(blob, nbytes, &c);
  if (st != PAI_OK) {
    fprintf(stderr, "inspect: %s\n", pai_status_str(st));
    free(blob);
    return 1;
  }

  pai_json_wb_init(&wb);
  pai_json_wb_puts(&wb, "{\"container\":{\"path\":");
  pai_json_quote(&wb, path, (uint32_t)strlen(path));
  pai_json_wb_putf(&wb, ",\"bytes\":%u,\"version\":%u,\"sections\":[",
                   c.nbytes, c.version);
  for (i = 0; i < c.num_sections; i++) {
    if (i) {
      pai_json_wb_puts(&wb, ",");
    }
    {
      const char *sn = pai_pai_section_name(c.sections[i].type);
      pai_json_wb_puts(&wb, "{\"type\":");
      pai_json_quote(&wb, sn, (uint32_t)strlen(sn));
      pai_json_wb_putf(&wb, ",\"offset\":%u,\"size\":%u}",
                       c.sections[i].offset, c.sections[i].size);
    }
  }
  pai_json_wb_puts(&wb, "]},\"meta\":");
  sec = pai_pai_section(&c, PAI_PAI_SEC_META, &sec_size);
  if (sec != NULL && pai_pai_meta_decode(sec, sec_size, &meta) == PAI_OK) {
    pai_json_wb_puts(&wb, "{\"name\":");
    pai_json_quote(&wb, meta.name, (uint32_t)strlen(meta.name));
    pai_json_wb_putf(&wb,
                     ",\"family\":%u,\"context_len\":%u,"
                     "\"num_layers\":%u,\"kv_bytes_per_token\":%u,"
                     "\"vocab_size\":%u}",
                     meta.family, meta.context_len, meta.num_layers,
                     meta.kv_bytes_per_token, meta.vocab_size);
  } else {
    pai_json_wb_puts(&wb, "null");
  }
  pai_json_wb_puts(&wb, ",\"tensors\":[");
  sec = pai_pai_section(&c, PAI_PAI_SEC_MANIFEST, &sec_size);
  if (sec != NULL &&
      pai_pai_manifest_decode(sec, sec_size, tensors, PAI_PAI_MAX_TENSORS,
                              &num_tensors) == PAI_OK) {
    for (i = 0; i < num_tensors; i++) {
      const pai_pai_tensor_t *t = &tensors[i];
      const char *dn = pai_dtype_name(t->dtype);
      if (i) {
        pai_json_wb_puts(&wb, ",");
      }
      pai_json_wb_puts(&wb, "{\"name\":");
      pai_json_quote(&wb, t->name, (uint32_t)strlen(t->name));
      pai_json_wb_putf(&wb, ",\"value_id\":%u,\"dtype\":", t->value_id);
      pai_json_quote(&wb, dn, (uint32_t)strlen(dn));
      pai_json_wb_putf(&wb, ",\"rank\":%u,\"shape\":", t->rank);
      json_shape(&wb, t->shape, t->rank);
      pai_json_wb_putf(&wb, ",\"offset\":%llu,\"size_bytes\":%llu}",
                       (unsigned long long)t->offset,
                       (unsigned long long)t->size_bytes);
    }
  }
  pai_json_wb_puts(&wb, "],\"ir\":");
  sec = pai_pai_section(&c, PAI_PAI_SEC_IR, &sec_size);
  if (sec != NULL && pai_ir_decode(sec, sec_size, &ir) == PAI_OK) {
    pai_json_wb_putf(&wb, "{\"values\":%u,\"ops\":%u,\"inputs\":%u,"
                          "\"outputs\":%u,\"ops\":[",
                     ir.num_values, ir.num_ops, ir.num_inputs,
                     ir.num_outputs);
    for (i = 1; i <= ir.num_ops; i++) {
      const pai_ir_op_t *o = &ir.ops[i];
      const char *kn = pai_ir_op_kind_name(o->kind);
      if (i > 1) {
        pai_json_wb_puts(&wb, ",");
      }
      pai_json_wb_putf(&wb, "{\"id\":%u,\"kind\":", i);
      pai_json_quote(&wb, kn, (uint32_t)strlen(kn));
      pai_json_wb_putf(&wb, ",\"inputs\":%u,\"outputs\":%u}",
                       o->num_inputs, o->num_outputs);
    }
    pai_json_wb_puts(&wb, "]}");
  } else {
    pai_json_wb_puts(&wb, "null");
  }
  pai_json_wb_puts(&wb, ",\"tokenizer\":");
  pai_tok_init(&tok);
  sec = pai_pai_section(&c, PAI_PAI_SEC_TOKENIZER, &sec_size);
  if (sec != NULL && pai_tok_deserialize(&tok, sec, sec_size) == PAI_OK) {
    pai_json_wb_putf(&wb, "{\"tokens\":%u,\"merges\":%u}",
                     tok.num_tokens, tok.num_merges);
  } else {
    pai_json_wb_puts(&wb, "null");
  }
  pai_tok_destroy(&tok);
  pai_json_wb_puts(&wb, "}\n");
  fwrite(wb.buf, 1, wb.len, stdout);
  pai_json_wb_destroy(&wb);
  free(blob);
  return 0;
}

/* ------------------------------------------------------------------ */
/* optimize (whitepaper §15/§20 mixed-precision planner)               */
/* ------------------------------------------------------------------ */

static void
fmt_bytes(char *out, size_t cap, uint64_t bytes) {
  if (bytes >= (UINT64_C(1) << 40)) {
    snprintf(out, cap, "%.2f TiB",
             (double)bytes / (double)(UINT64_C(1) << 40));
  } else if (bytes >= (UINT64_C(1) << 30)) {
    snprintf(out, cap, "%.2f GiB",
             (double)bytes / (double)(UINT64_C(1) << 30));
  } else if (bytes >= (UINT64_C(1) << 20)) {
    snprintf(out, cap, "%.2f MiB",
             (double)bytes / (double)(UINT64_C(1) << 20));
  } else if (bytes >= (UINT64_C(1) << 10)) {
    snprintf(out, cap, "%.2f KiB",
             (double)bytes / (double)(UINT64_C(1) << 10));
  } else {
    snprintf(out, cap, "%llu B", (unsigned long long)bytes);
  }
}

/* Parse a --budget value: plain bytes, K/M/G/T suffix, or NN% of the
 * current total. Returns 0 with *ok = 0 on malformed input. */
static uint64_t
parse_budget(const char *s, uint64_t cur_total, int *ok) {
  char *end;
  double v;
  uint64_t mult;

  *ok = 0;
  if (s == NULL || s[0] == '\0') {
    return 0;
  }
  v = strtod(s, &end);
  if (end == s || v < 0) {
    return 0;
  }
  if (*end == '%') {
    if (end[1] != '\0') {
      return 0;
    }
    *ok = 1;
    return (uint64_t)(v * 0.01 * (double)cur_total);
  }
  /* Binary units, consistent with the KiB/MiB report (fmt_bytes). */
  mult = 1;
  if (*end == 'K' || *end == 'k') {
    mult = UINT64_C(1) << 10;
    end++;
  } else if (*end == 'M' || *end == 'm') {
    mult = UINT64_C(1) << 20;
    end++;
  } else if (*end == 'G' || *end == 'g') {
    mult = UINT64_C(1) << 30;
    end++;
  } else if (*end == 'T' || *end == 't') {
    mult = UINT64_C(1) << 40;
    end++;
  }
  if (*end != '\0') {
    return 0;
  }
  *ok = 1;
  return (uint64_t)(v * (double)mult);
}

static int
cmd_optimize(int argc, char **argv) {
  const char *path;
  const char *budget_str = NULL;
  uint16_t group = 0;
  int json = 0;
  uint8_t *blob = NULL;
  uint32_t nbytes = 0;
  pai_pai_container_t c;
  pai_ir_program_t ir;
  int have_ir = 0;
  pai_opt_plan_t plan;
  pai_status_t st;
  int i;

  if (argc < 3) {
    usage();
    return 2;
  }
  path = argv[2];
  for (i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--budget") == 0 && i + 1 < argc) {
      budget_str = argv[++i];
    } else if (strcmp(argv[i], "--group") == 0 && i + 1 < argc) {
      group = (uint16_t)strtoul(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "--json") == 0) {
      json = 1;
    } else {
      usage();
      return 2;
    }
  }

  st = pai_pai_read_file(path, &blob, &nbytes);
  if (st != PAI_OK) {
    fprintf(stderr, "optimize: %s\n", pai_status_str(st));
    return 1;
  }
  st = pai_pai_open(blob, nbytes, &c);
  if (st != PAI_OK) {
    fprintf(stderr, "optimize: %s\n", pai_status_str(st));
    free(blob);
    return 1;
  }
  {
    const uint8_t *sec;
    uint32_t sec_size;
    sec = pai_pai_section(&c, PAI_PAI_SEC_IR, &sec_size);
    if (sec != NULL && pai_ir_decode(sec, sec_size, &ir) == PAI_OK) {
      have_ir = 1;
    }
  }

  /* First pass without a budget to learn the current total (percent
   * budgets need it), then re-plan with the real budget. */
  st = pai_opt_plan(&c, have_ir ? &ir : NULL, 0, group, &plan);
  if (st != PAI_OK) {
    fprintf(stderr, "optimize: %s\n", pai_status_str(st));
    free(blob);
    return 1;
  }
  if (budget_str != NULL) {
    int ok = 0;
    uint64_t budget = parse_budget(budget_str, plan.cur_total, &ok);
    if (!ok) {
      fprintf(stderr, "optimize: invalid --budget '%s'\n", budget_str);
      free(blob);
      return 2;
    }
    st = pai_opt_plan(&c, have_ir ? &ir : NULL, budget, group, &plan);
    if (st != PAI_OK) {
      fprintf(stderr, "optimize: %s\n", pai_status_str(st));
      free(blob);
      return 1;
    }
  }

  if (json) {
    pai_json_wb_t wb;
    uint32_t t;
    pai_json_wb_init(&wb);
    pai_json_wb_puts(&wb, "{\"model\":");
    pai_json_quote(&wb, plan.name, (uint32_t)strlen(plan.name));
    pai_json_wb_putf(&wb, ",\"current_bytes\":%llu,\"plan_bytes\":%llu",
                     (unsigned long long)plan.cur_total,
                     (unsigned long long)plan.plan_total);
    if (plan.budget != 0) {
      pai_json_wb_putf(&wb, ",\"budget\":%llu",
                       (unsigned long long)plan.budget);
    } else {
      pai_json_wb_puts(&wb, ",\"budget\":null");
    }
    pai_json_wb_putf(&wb, ",\"min_bytes\":%llu,\"feasible\":%s",
                     (unsigned long long)plan.min_total,
                     plan.feasible ? "true" : "false");
    pai_json_wb_puts(&wb, ",\"tensors\":[");
    for (t = 0; t < plan.num_tensors; t++) {
      const pai_opt_tensor_t *o = &plan.tensors[t];
      if (t) {
        pai_json_wb_puts(&wb, ",");
      }
      pai_json_wb_puts(&wb, "{\"name\":");
      pai_json_quote(&wb, o->name, (uint32_t)strlen(o->name));
      pai_json_wb_puts(&wb, ",\"role\":");
      {
        const char *rn = pai_opt_role_name(o->role);
        pai_json_quote(&wb, rn, (uint32_t)strlen(rn));
      }
      pai_json_wb_putf(&wb, ",\"value_id\":%u,\"numel\":%llu",
                       o->value_id, (unsigned long long)o->numel);
      pai_json_wb_puts(&wb, ",\"current_scheme\":");
      {
        const char *sn = pai_opt_scheme_name(o->cur_scheme);
        pai_json_quote(&wb, sn, (uint32_t)strlen(sn));
      }
      pai_json_wb_putf(&wb, ",\"current_bytes\":%llu",
                       (unsigned long long)o->cur_bytes);
      pai_json_wb_puts(&wb, ",\"plan_scheme\":");
      {
        const char *sn = pai_opt_scheme_name(o->chosen);
        pai_json_quote(&wb, sn, (uint32_t)strlen(sn));
      }
      pai_json_wb_putf(&wb, ",\"plan_bytes\":%llu}",
                       (unsigned long long)o->size_bytes[o->chosen]);
    }
    pai_json_wb_puts(&wb, "]}\n");
    fwrite(wb.buf, 1, wb.len, stdout);
    pai_json_wb_destroy(&wb);
  } else {
    char cur[48];
    char pln[48];
    double pct;
    uint32_t t;
    pct = plan.cur_total > 0
              ? 100.0 * (1.0 - (double)plan.plan_total / (double)plan.cur_total)
              : 0.0;
    printf("optimize: %s\n", plan.name);
    fmt_bytes(cur, sizeof(cur), plan.cur_total);
    printf("  current %s\n", cur);
    fmt_bytes(pln, sizeof(pln), plan.plan_total);
    printf("  plan    %s  (-%.1f%%)\n", pln, pct);
    if (plan.budget != 0) {
      char bd[48];
      char mn[48];
      fmt_bytes(bd, sizeof(bd), plan.budget);
      fmt_bytes(mn, sizeof(mn), plan.min_total);
      printf("  budget  %s (%s; min %s)\n", bd,
             plan.feasible ? "feasible" : "INFEASIBLE — best effort", mn);
    }
    printf("  %-30s %-10s %12s %12s %10s\n", "tensor", "role",
           "current", "plan", "saving");
    for (t = 0; t < plan.num_tensors; t++) {
      const pai_opt_tensor_t *o = &plan.tensors[t];
      char cb[32];
      char pb[32];
      char sv[32];
      fmt_bytes(cb, sizeof(cb), o->cur_bytes);
      fmt_bytes(pb, sizeof(pb), o->size_bytes[o->chosen]);
      fmt_bytes(sv, sizeof(sv),
                o->cur_bytes > o->size_bytes[o->chosen]
                    ? o->cur_bytes - o->size_bytes[o->chosen]
                    : 0);
      printf("  %-30s %-10s %12s %12s %10s\n", o->name,
             pai_opt_role_name(o->role), cb, pb, sv);
    }
  }

  free(blob);
  return 0;
}

/* ------------------------------------------------------------------ */
/* validate                                                            */
/* ------------------------------------------------------------------ */

static int
cmd_validate(int argc, char **argv) {
  const char *path;
  uint8_t *blob = NULL;
  uint32_t nbytes = 0;
  pai_pai_container_t c;
  pai_model_t *model = NULL;
  pai_status_t st;

  if (argc < 3) {
    usage();
    return 2;
  }
  path = argv[2];

  st = pai_pai_read_file(path, &blob, &nbytes);
  if (st != PAI_OK) {
    fprintf(stderr, "validate: %s\n", pai_status_str(st));
    return 1;
  }
  st = pai_pai_open(blob, nbytes, &c);
  if (st != PAI_OK) {
    fprintf(stderr, "validate: container invalid: %s\n", pai_status_str(st));
    free(blob);
    return 1;
  }
  /* Full load: manifest/weights/IR/tokenizer consistency + memory plan. */
  st = pai_model_open_blob(blob, nbytes, &model);
  if (st != PAI_OK) {
    fprintf(stderr, "validate: model invalid: %s\n", pai_status_str(st));
    free(blob);
    return 1;
  }
  pai_model_close(model);
  free(blob);
  printf("validate: %s VALID\n", path);
  return 0;
}

/* ------------------------------------------------------------------ */
/* benchmark                                                           */
/* ------------------------------------------------------------------ */

static int
cmd_benchmark(int argc, char **argv) {
  const char *path;
  const char *prompt = "a";
  uint32_t steps = 8;
  uint32_t iters = 10;
  pai_model_t *model = NULL;
  pai_session_t *session = NULL;
  pai_status_t st;
  int i;

  for (i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
      steps = (uint32_t)strtoul(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
      prompt = argv[++i];
    } else {
      usage();
      return 2;
    }
  }
  if (argc < 3) {
    usage();
    return 2;
  }
  path = argv[2];
  if (steps == 0) {
    steps = 8;
  }

  st = pai_model_open(NULL, path, &model);
  if (st != PAI_OK) {
    fprintf(stderr, "benchmark: %s\n", pai_status_str(st));
    return 1;
  }
  st = pai_session_create(model, &session);
  if (st != PAI_OK) {
    fprintf(stderr, "benchmark: %s\n", pai_status_str(st));
    pai_model_close(model);
    return 1;
  }
  st = pai_session_set_generation(session, steps, 0);
  if (st != PAI_OK) {
    fprintf(stderr, "benchmark: %s\n", pai_status_str(st));
    pai_session_destroy(session);
    pai_model_close(model);
    return 1;
  }

  {
    uint64_t *samples = (uint64_t *)malloc(iters * sizeof(uint64_t));
    if (samples == NULL) {
      return 2;
    }
    /* Warmup. */
    for (i = 0; i < 1; i++) {
      pai_generate(session, prompt, NULL, NULL);
    }
    for (i = 0; i < (int)iters; i++) {
      uint64_t t0 = pai_bench_now_ns();
      st = pai_generate(session, prompt, NULL, NULL);
      samples[i] = pai_bench_now_ns() - t0;
      if (st != PAI_OK) {
        fprintf(stderr, "benchmark: generate failed: %s\n",
                pai_status_str(st));
        free(samples);
        pai_session_destroy(session);
        pai_model_close(model);
        return 1;
      }
    }
    {
      uint64_t med = pai_bench_median(samples, iters);
      uint64_t mn = pai_bench_min(samples, iters);
      double tokens = (double)steps;
      printf("model %s | %u tokens/gen | min %.3f ms (%.1f tok/s) | median "
             "%.3f ms (%.1f tok/s)\n",
             pai_model_name(model), steps, (double)mn / 1e6,
             tokens * 1e9 / (double)mn, (double)med / 1e6,
             tokens * 1e9 / (double)med);
    }
    free(samples);
  }

  pai_session_destroy(session);
  pai_model_close(model);
  return 0;
}

/* ------------------------------------------------------------------ */
/* proto-ping (whitepaper §24/§25 connectivity check)                  */
/* ------------------------------------------------------------------ */

static void
print_caps(uint32_t caps) {
  static const struct {
    uint32_t bit;
    const char *name;
  } k_caps[] = {
      {PAI_PROTO_CAP_STREAMS, "streams"},     {PAI_PROTO_CAP_SESSIONS, "sessions"},
      {PAI_PROTO_CAP_GENERATE, "generate"},   {PAI_PROTO_CAP_EMBED, "embed"},
      {PAI_PROTO_CAP_TRANSFER, "transfer"},   {PAI_PROTO_CAP_TELEMETRY, "telemetry"},
      {PAI_PROTO_CAP_COMPRESSION, "compression"},
  };
  int first = 1;
  for (size_t i = 0; i < sizeof(k_caps) / sizeof(k_caps[0]); i++) {
    if (caps & k_caps[i].bit) {
      printf("%s%s", first ? "" : ", ", k_caps[i].name);
      first = 0;
    }
  }
  if (first) {
    printf("(none)");
  }
}

static int
cmd_proto_ping(int argc, char **argv) {
  const char *host;
  uint16_t port;
  uint32_t count = 5;
  uint32_t caps = 0;
  pai_proto_ping_result_t results[64];
  pai_status_t st;
  int i;

  if (argc < 4) {
    usage();
    return 2;
  }
  host = argv[2];
  {
    unsigned long p = strtoul(argv[3], NULL, 10);
    if (p < 1 || p > 65535) {
      fprintf(stderr, "proto-ping: invalid port\n");
      return 2;
    }
    port = (uint16_t)p;
  }
  for (i = 4; i < argc; i++) {
    if (strcmp(argv[i], "--count") == 0 && i + 1 < argc) {
      count = (uint32_t)strtoul(argv[++i], NULL, 10);
    } else {
      usage();
      return 2;
    }
  }
  if (count == 0 || count > 64) {
    fprintf(stderr, "proto-ping: count must be 1..64\n");
    return 2;
  }

  st = pai_proto_ping(host, port, count, results, &caps, 2000);
  if (st != PAI_OK) {
    fprintf(stderr, "proto-ping: %s:%u: %s\n", host, (unsigned)port,
            pai_status_str(st));
    return 1;
  }

  printf("proto-ping %s:%u — %u ping(s)\n", host, (unsigned)port, count);
  printf("  negotiated caps 0x%08x (", caps);
  print_caps(caps);
  printf(")\n");

  {
    uint64_t *samples = (uint64_t *)malloc(count * sizeof(uint64_t));
    uint32_t ns = 0;
    uint32_t lost = 0;
    uint64_t mn = 0;
    uint64_t mx = 0;
    uint64_t med = 0;
    if (samples == NULL) {
      return 2;
    }
    for (i = 0; i < (int)count; i++) {
      if (results[i].lost) {
        printf("  #%-2d  timeout\n", i + 1);
        lost++;
      } else {
        printf("  #%-2d  rtt %.3f ms\n", i + 1,
               (double)results[i].rtt_ns / 1e6);
        samples[ns++] = results[i].rtt_ns;
        if (ns == 1) {
          mn = mx = results[i].rtt_ns;
        } else {
          if (results[i].rtt_ns < mn) {
            mn = results[i].rtt_ns;
          }
          if (results[i].rtt_ns > mx) {
            mx = results[i].rtt_ns;
          }
        }
      }
    }
    if (ns > 0) {
      med = pai_bench_median(samples, ns);
    }
    if (ns > 0) {
      printf("  min %.3f ms | median %.3f ms | max %.3f ms | %u/%u lost\n",
             (double)mn / 1e6, (double)med / 1e6, (double)mx / 1e6, lost,
             count);
    } else {
      printf("  %u/%u lost — peer unreachable or unresponsive\n", lost,
             count);
    }
    free(samples);
    /* All pings lost: report failure to scripts even though the TCP
     * connection and negotiation succeeded. */
    return lost == count ? 1 : 0;
  }
}

/* ------------------------------------------------------------------ */
/* serve (whitepaper §26 OpenAI-compatible gateway)                    */
/* ------------------------------------------------------------------ */

static pai_status_t
gw_http_handler(void *user, const pai_http_req_t *req,
                pai_http_resp_t *resp) {
  return pai_gw_handle_request((pai_gw_t *)user, req->method, req->target,
                               req->body, req->body_len, resp);
}

static int
cmd_serve(int argc, char **argv) {
  const char *host = "127.0.0.1";
  uint16_t port = 8080;
  pai_gw_t gw;
  pai_http_server_t server;
  pai_status_t st;
  int i;
  int nmodels = 0;

  for (i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
      host = argv[++i];
    } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
      long p = strtol(argv[++i], NULL, 10);
      if (p < 1 || p > 65535) {
        fprintf(stderr, "serve: invalid port\n");
        return 2;
      }
      port = (uint16_t)p;
    } else if (strcmp(argv[i], "--remote") == 0 && i + 1 < argc) {
      /* Remote payload model: [name@]host:port (Prospero Protocol). */
      const char *spec = argv[++i];
      const char *at = strchr(spec, '@');
      const char *colon = strrchr(spec, ':');
      char name[256];
      char hostbuf[256];
      long p;
      size_t name_n, host_n;

      if (nmodels == 0) {
        st = pai_gw_init(&gw);
        if (st != PAI_OK) {
          fprintf(stderr, "serve: %s\n", pai_status_str(st));
          return 1;
        }
      }
      if (colon == NULL) {
        fprintf(stderr, "serve: --remote wants [name@]host:port\n");
        pai_gw_destroy(&gw);
        return 2;
      }
      {
        char *endp = NULL;
        p = strtol(colon + 1, &endp, 10);
        if (endp == colon + 1 || *endp != '\0' || p < 1 || p > 65535) {
          fprintf(stderr, "serve: --remote: invalid port\n");
          pai_gw_destroy(&gw);
          return 2;
        }
      }
      host_n = (size_t)(colon - (at != NULL ? at + 1 : spec));
      if (host_n == 0 || host_n >= sizeof(hostbuf)) {
        fprintf(stderr, "serve: --remote: invalid host\n");
        pai_gw_destroy(&gw);
        return 2;
      }
      memcpy(hostbuf, at != NULL ? at + 1 : spec, host_n);
      hostbuf[host_n] = '\0';
      if (at != NULL) {
        name_n = (size_t)(at - spec);
        if (name_n == 0 || name_n >= sizeof(name)) {
          fprintf(stderr, "serve: --remote: invalid name\n");
          pai_gw_destroy(&gw);
          return 2;
        }
        memcpy(name, spec, name_n);
        name[name_n] = '\0';
      } else {
        if (strlen(hostbuf) >= sizeof(name)) {
          fprintf(stderr, "serve: --remote: invalid name\n");
          pai_gw_destroy(&gw);
          return 2;
        }
        strcpy(name, hostbuf); /* default registry id: the host */
      }
      st = pai_gw_add_remote(&gw, name, hostbuf, (uint16_t)p);
      if (st != PAI_OK) {
        fprintf(stderr, "serve: remote %s: %s\n", spec, pai_status_str(st));
        pai_gw_destroy(&gw);
        return 1;
      }
      nmodels++;
    } else if (argv[i][0] != '-') {
      /* Model path. */
      if (nmodels == 0) {
        st = pai_gw_init(&gw);
        if (st != PAI_OK) {
          fprintf(stderr, "serve: %s\n", pai_status_str(st));
          return 1;
        }
      }
      st = pai_gw_add_file(&gw, argv[i]);
      if (st != PAI_OK) {
        fprintf(stderr, "serve: %s: %s\n", argv[i], pai_status_str(st));
        pai_gw_destroy(&gw);
        return 1;
      }
      nmodels++;
    } else {
      usage();
      return 2;
    }
  }
  if (nmodels == 0) {
    usage();
    return 2;
  }

  st = pai_http_server_init(&server, host, port, gw_http_handler, &gw);
  if (st != PAI_OK) {
    fprintf(stderr, "serve: cannot bind %s:%u: %s\n", host, (unsigned)port,
            pai_status_str(st));
    pai_gw_destroy(&gw);
    return 1;
  }
  printf("serving %d model(s) on http://%s:%u\n", nmodels, host,
         (unsigned)port);
  printf("  GET  /v1/models\n");
  printf("  POST /v1/completions, /v1/chat/completions, /v1/embeddings\n");
  fflush(stdout);

  st = pai_http_server_run(&server);
  pai_http_server_close(&server);
  pai_gw_destroy(&gw);
  return st == PAI_OK ? 0 : 1;
}

int
main(int argc, char **argv) {
  int i;
  int json_mode = 0;

  if (argc < 2) {
    usage();
    return 2;
  }
  /* Machine-readable modes must keep stdout pure JSON: the version
   * banner is skipped whenever --json is present. */
  for (i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--json") == 0) {
      json_mode = 1;
    }
  }
  if (!json_mode) {
    printf("pai (ProsperoAI %s, %s)\n", pai_version_string(),
           PAI_MILESTONE);
  }
  if (strcmp(argv[1], "convert") == 0) {
    return cmd_convert(argc, argv);
  }
  if (strcmp(argv[1], "inspect") == 0) {
    return cmd_inspect(argc, argv);
  }
  if (strcmp(argv[1], "validate") == 0) {
    return cmd_validate(argc, argv);
  }
  if (strcmp(argv[1], "benchmark") == 0) {
    return cmd_benchmark(argc, argv);
  }
  if (strcmp(argv[1], "optimize") == 0) {
    return cmd_optimize(argc, argv);
  }
  if (strcmp(argv[1], "serve") == 0) {
    return cmd_serve(argc, argv);
  }
  if (strcmp(argv[1], "proto-ping") == 0) {
    return cmd_proto_ping(argc, argv);
  }
  fprintf(stderr, "unknown command: %s\n", argv[1]);
  usage();
  return 2;
}
