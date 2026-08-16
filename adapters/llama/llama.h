/*
 * ProsperoAI — LLaMA-family adapter (whitepaper §9.3).
 * Translates a llama.cpp GGUF file (LLaMA-2/3, Mistral) into a native
 * `.pai` container: faithful LLaMA graph (RMSNorm, QKV projections,
 * causal multi-head attention, RoPE, SiLU-gated MLP, residuals, LM
 * head) -> optional quantization (§15) -> PAI packaging (§20).
 * v0 runs in sequence mode, recomputing attention over the full
 * context each step (O(seq^2); incremental KV cache is a later
 * milestone). Host-side tooling only.
 */

#ifndef PAI_ADAPTERS_LLAMA_H
#define PAI_ADAPTERS_LLAMA_H

#include <container.h>

#include <graph/graph.h>

#include <pai/error.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Architecture families accepted by the v0 translator. */
#define PAI_LLAMA_ARCH_LLAMA "llama"
#define PAI_LLAMA_ARCH_MISTRAL "mistral"

/*
 * Import a GGUF file into a `.pai` container at `out_path`.
 * `quant` may be NULL (keep f32 weights). The model metadata, graph
 * and tokenizer come from the GGUF metadata + tensors.
 */
pai_status_t pai_llama_import(const char *gguf_path, const char *out_path,
                              const pai_container_quant_t *quant);

/* True when the first 4 bytes look like a GGUF file. */
int pai_llama_is_gguf(const uint8_t *head4);

#ifdef __cplusplus
}
#endif

#endif /* PAI_ADAPTERS_LLAMA_H */
