/* Shared `.pai` container builder (see container.h): one packaging
 * path for the DSL importer and the GGUF adapter. */

#include "container.h"

#include <graph/graph.h>
#include <ir/ir.h>

#include <quantize.h>
#include <tokenizer.h>

#include <stdlib.h>
#include <string.h>

static uint64_t
tensor_numel(const pai_container_tensor_t *t) {
  uint64_t n = 1;
  for (uint32_t i = 0; i < t->rank; i++) {
    n *= t->shape[i];
  }
  return n;
}

pai_status_t
pai_container_build(const pai_pai_meta_t *meta,
                    const pai_container_tensor_t *tensors, uint32_t num_tensors,
                    const pai_container_op_t *ops, uint32_t num_ops,
                    const pai_container_token_t *tokens, uint32_t num_tokens,
                    const pai_container_merge_t *merges, uint32_t num_merges,
                    const pai_container_quant_t *quant,
                    uint8_t **out_blob, uint32_t *out_nbytes) {
  pai_graph_t g;
  pai_ir_program_t ir;
  pai_status_t st;
  pai_tok_t tok;
  int quant_on = 0;
  int quant_q4 = 0;
  uint8_t *ir_blob = NULL;
  uint32_t ir_n = 0;
  uint8_t *tok_blob = NULL;
  uint32_t tok_n = 0;
  uint8_t meta_blob[PAI_PAI_META_SIZE];
  uint32_t meta_n = 0;
  pai_pai_tensor_t *tensor_meta = NULL;
  uint8_t *manifest_blob = NULL;
  uint32_t manifest_n = 0;
  uint32_t num_manifest = 0;
  uint8_t *weights = NULL;
  uint64_t weights_n = 0;
  pai_pai_builder_t builder;
  uint8_t *container = NULL;
  uint32_t total = 0;
  uint32_t i;

  if (out_blob == NULL || out_nbytes == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  *out_blob = NULL;
  *out_nbytes = 0;
  if (meta == NULL || tensors == NULL || num_tensors == 0 ||
      (ops == NULL && num_ops > 0) || (tokens == NULL && num_tokens > 0) ||
      (merges == NULL && num_merges > 0)) {
    return PAI_ERR_INVALID_ARG;
  }

  if (quant != NULL && quant->quant != NULL &&
      strcmp(quant->quant, "none") != 0) {
    if (strcmp(quant->quant, "q8") != 0 && strcmp(quant->quant, "q4") != 0) {
      return PAI_ERR_INVALID_ARG; /* reject bad schemes up front        */
    }
    quant_on = 1;
    quant_q4 = strcmp(quant->quant, "q4") == 0;
  }

  pai_tok_init(&tok);

  /* Graph reconstruction: value ids are 1..N in declaration order. */
  pai_graph_init(&g);
  for (i = 0; i < num_tensors; i++) {
    const pai_container_tensor_t *t = &tensors[i];
    pai_graph_value_id id;
    uint64_t n = tensor_numel(t);

    if (t->id != i + 1 || t->rank == 0 || t->rank > PAI_TENSOR_MAX_RANK ||
        n == 0) {
      st = PAI_ERR_INVALID_ARG;
      goto done;
    }
    if (t->kind == 1 && (t->name == NULL || t->data == NULL ||
                         t->n == 0 || t->n != n)) {
      st = PAI_ERR_INVALID_ARG; /* params need weights + manifest name */
      goto done;
    }
    id = pai_graph_add_value(&g, (pai_dtype_t)t->dtype, t->rank, t->shape,
                             16);
    if (id != t->id) {
      st = PAI_ERR_MISMATCH;
      goto done;
    }
    if (t->kind == 0) {
      pai_graph_set_input(&g, id);
    } else if (t->kind == 3) {
      pai_graph_set_output(&g, id);
    }
  }
  for (i = 0; i < num_ops; i++) {
    const pai_container_op_t *o = &ops[i];
    pai_graph_op_id oid;

    if (o->id != i + 1 || o->num_in == 0 || o->num_out == 0 ||
        o->num_in > PAI_GRAPH_MAX_ARITY || o->num_out > PAI_GRAPH_MAX_ARITY ||
        o->in == NULL || o->out == NULL) {
      st = PAI_ERR_INVALID_ARG;
      goto done;
    }
    oid = pai_graph_add_op(&g, (pai_graph_op_kind_t)o->kind, o->num_in,
                           o->num_out, o->in, o->out);
    if (oid != o->id) {
      st = PAI_ERR_MISMATCH;
      goto done;
    }
  }
  st = pai_graph_topo_sort(&g);
  if (st != PAI_OK) {
    goto done;
  }

  /* Prospero IR. */
  st = pai_ir_from_graph(&g, &ir);
  if (st != PAI_OK) {
    goto done;
  }

  /* Quantization metadata (§15) for the packed params. */
  if (quant_on) {
    pai_ir_quant_t q;
    memset(&q, 0, sizeof(q));
    q.present = 1;
    q.bit_width = quant_q4 ? 4 : 8;
    q.scale_repr = PAI_IR_QUANT_REPR_F32;
    q.zero_point_repr = PAI_IR_QUANT_REPR_F32;
    q.is_signed = 1;
    q.group_size = quant->quant_group;
    q.block_structure = q.group_size == 0 ? PAI_IR_QUANT_BLOCK_TENSOR
                                          : PAI_IR_QUANT_BLOCK_GROUP;
    for (i = 0; i < num_tensors; i++) {
      if (tensors[i].kind == 1 && tensors[i].data != NULL &&
          tensors[i].quantize) {
        st = pai_ir_value_set_quant(&ir, tensors[i].id, &q);
        if (st != PAI_OK) {
          goto done;
        }
      }
    }
  }

  /* Weights section + manifest. */
  tensor_meta = (pai_pai_tensor_t *)calloc(num_tensors, sizeof(*tensor_meta));
  if (tensor_meta == NULL) {
    st = PAI_ERR_NOMEM;
    goto done;
  }
  for (i = 0; i < num_tensors; i++) {
    const pai_container_tensor_t *t = &tensors[i];
    pai_pai_tensor_t *tm;
    uint64_t n;
    uint64_t size;

    if (t->kind != 1 || t->data == NULL) {
      continue;
    }
    n = t->n;
    size = n * sizeof(float);
    if (quant_on && t->quantize) {
      pai_quant_scheme_t scheme;
      scheme.bit_width = quant_q4 ? 4 : 8;
      scheme.is_signed = 1;
      scheme.group_size = quant->quant_group;
      size = pai_quant_weights_bytes(n, &scheme);
    }

    {
      uint8_t *nw = (uint8_t *)realloc(weights, (size_t)(weights_n + size));
      if (nw == NULL) {
        st = PAI_ERR_NOMEM;
        goto done;
      }
      weights = nw;
    }

    tm = &tensor_meta[num_manifest];
    memset(tm, 0, sizeof(*tm));
    strncpy(tm->name, t->name, sizeof(tm->name) - 1);
    tm->value_id = t->id;
    tm->dtype = PAI_DTYPE_F32;
    tm->rank = t->rank;
    memcpy(tm->shape, t->shape, sizeof(tm->shape));
    tm->offset = weights_n;
    tm->size_bytes = size;

    if (!quant_on || !t->quantize) {
      memcpy(weights + weights_n, t->data, (size_t)size);
    } else {
      pai_quant_scheme_t scheme;
      int8_t *q;
      float *scales;
      uint64_t q_bytes;
      uint64_t sc_bytes;

      scheme.bit_width = quant_q4 ? 4 : 8;
      scheme.is_signed = 1;
      scheme.group_size = quant->quant_group;
      q_bytes = pai_quant_value_bytes(n, &scheme);
      sc_bytes = pai_quant_scale_bytes(n, &scheme);
      q = (int8_t *)malloc((size_t)q_bytes);
      scales = (float *)malloc((size_t)sc_bytes);
      if (q == NULL || scales == NULL) {
        free(q);
        free(scales);
        st = PAI_ERR_NOMEM;
        goto done;
      }
      st = pai_quantize_f32(t->data, n, &scheme, q, scales, NULL);
      if (st != PAI_OK) {
        free(q);
        free(scales);
        goto done;
      }
      memcpy(weights + weights_n, q, (size_t)q_bytes);
      memcpy(weights + weights_n + q_bytes, scales, (size_t)sc_bytes);
      free(q);
      free(scales);
    }
    weights_n += size;
    num_manifest++;
  }

  /* Tokenizer. */
  for (i = 0; i < num_tokens; i++) {
    st = pai_tok_add_token(&tok, tokens[i].id, tokens[i].text);
    if (st != PAI_OK) {
      goto done;
    }
  }
  for (i = 0; i < num_merges; i++) {
    st = pai_tok_add_merge(&tok, merges[i].left, merges[i].right,
                           merges[i].result);
    if (st != PAI_OK) {
      goto done;
    }
  }
  pai_tok_finalize(&tok);

  /* Metadata. */
  st = pai_pai_meta_encode(meta, meta_blob, sizeof(meta_blob), &meta_n);
  if (st != PAI_OK) {
    goto done;
  }
  manifest_blob = (uint8_t *)malloc(8u + num_manifest * 160u);
  if (manifest_blob == NULL) {
    st = PAI_ERR_NOMEM;
    goto done;
  }
  st = pai_pai_manifest_encode(tensor_meta, num_manifest, manifest_blob,
                               8u + num_manifest * 160u, &manifest_n);
  if (st != PAI_OK) {
    goto done;
  }

  /* IR blob. */
  ir_n = pai_ir_encoded_size(&ir);
  if (ir_n == 0) {
    st = PAI_ERR_MISMATCH;
    goto done;
  }
  ir_blob = (uint8_t *)malloc(ir_n);
  if (ir_blob == NULL) {
    st = PAI_ERR_NOMEM;
    goto done;
  }
  st = pai_ir_encode(&ir, ir_blob, ir_n, &ir_n);
  if (st != PAI_OK) {
    goto done;
  }

  /* Tokenizer blob. */
  tok_n = pai_tok_blob_size(&tok);
  tok_blob = (uint8_t *)malloc(tok_n > 0 ? tok_n : 1);
  if (tok_blob == NULL) {
    st = PAI_ERR_NOMEM;
    goto done;
  }
  if (tok_n > 0) {
    st = pai_tok_serialize(&tok, tok_blob, tok_n, &tok_n);
    if (st != PAI_OK) {
      goto done;
    }
  }

  /* Container. */
  pai_pai_builder_init(&builder);
  pai_pai_builder_add(&builder, PAI_PAI_SEC_META, meta_blob, meta_n);
  pai_pai_builder_add(&builder, PAI_PAI_SEC_MANIFEST, manifest_blob,
                      manifest_n);
  pai_pai_builder_add(&builder, PAI_PAI_SEC_WEIGHTS, weights,
                      (uint32_t)weights_n);
  pai_pai_builder_add(&builder, PAI_PAI_SEC_IR, ir_blob, ir_n);
  if (tok_n > 0) {
    pai_pai_builder_add(&builder, PAI_PAI_SEC_TOKENIZER, tok_blob, tok_n);
  }
  total = pai_pai_encoded_size(&builder);
  if (total == 0) {
    st = PAI_ERR_MISMATCH;
    goto done;
  }
  container = (uint8_t *)malloc(total);
  if (container == NULL) {
    st = PAI_ERR_NOMEM;
    goto done;
  }
  st = pai_pai_build(&builder, container, total, &total);
  if (st != PAI_OK) {
    goto done;
  }

  *out_blob = container;
  *out_nbytes = total;
  container = NULL; /* ownership transferred */
  st = PAI_OK;

done:
  free(container);
  free(tok_blob);
  free(ir_blob);
  free(manifest_blob);
  free(tensor_meta);
  free(weights);
  pai_tok_destroy(&tok);
  return st;
}
