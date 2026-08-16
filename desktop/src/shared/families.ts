/*
 * ProsperoAI Desktop — model family detection (shared main/renderer).
 *
 * Real sources feed this: Hugging Face search tags carry the model_type
 * token ("qwen2", "llama", "gemma2", "phi3"...), Ollama's /api/tags
 * exposes details.family, and model ids embed the family name
 * ("Llama-3.1-8B", "Qwen2.5-0.5B"). All of them are normalised here to
 * a canonical family id that the renderer maps to a logo + brand colors.
 */

const FAMILY_ALIASES: Record<string, string> = {
  /* exact-token aliases (checked before digit stripping) */
  'gpt2': 'gpt',
  'gptj': 'gpt',
  'gptneo': 'gpt',
  'gptneox': 'gpt',
  'gpt4': 'gpt',
  'chatgpt': 'gpt',
  'openai': 'gpt',
  'o1': 'gpt',
  'o3': 'gpt',
  't5': 't5',
  'flan': 't5',
  'roberta': 'bert',
  'electra': 'bert',
  'mixtral': 'mistral',
  'codestral': 'mistral',
  'ministral': 'mistral',
  'mathstral': 'mistral',
  'devstral': 'mistral',
  'commandr': 'command',
  'starcoder2': 'starcoder',
  'solar': 'solar',
  'dbrx': 'dbrx',
  'mpt': 'mpt',
  'glm': 'glm',
  'internlm': 'internlm',
  'jamba': 'jamba',
  'olmo': 'olmo',
  'yi': 'yi',
  'grok1': 'grok',
  'grok2': 'grok',
  'grok3': 'grok',
};

const KNOWN_FAMILIES = new Set([
  'llama', 'qwen', 'mistral', 'gemma', 'phi', 'deepseek', 'falcon', 'gpt',
  'bert', 't5', 'yi', 'grok', 'command', 'granite', 'nemotron', 'olmo',
  'starcoder', 'solar', 'mpt', 'dbrx', 'glm', 'internlm', 'jamba',
]);

/** Normalise a single token (model_type tag, Ollama family, id fragment). */
export function normalizeFamily(raw: string): string | undefined {
  const token = raw.toLowerCase().replace(/[^a-z0-9]+/g, '');
  if (!token) return undefined;
  const alias = FAMILY_ALIASES[token];
  if (alias) return alias;
  /* Family tokens routinely carry a version suffix: qwen2 -> qwen,
   * gemma2 -> gemma, llama3_1 -> llama, deepseek_v2 -> deepseek. */
  const stripped = token.replace(/[0-9]+$/, '');
  if (stripped && KNOWN_FAMILIES.has(stripped)) return stripped;
  if (KNOWN_FAMILIES.has(token)) return token;
  /* Suffixes that survive stripping ("llama3.2:3b" -> "llama323b",
   * "deepseek-coder" -> "deepseekcoder"): match a known family as a
   * prefix, preferring the longest one. Short families are skipped
   * except t5, which only appears in real T5 identifiers. */
  if (token.length >= 4) {
    let best: string | undefined;
    for (const family of KNOWN_FAMILIES) {
      const usable = family.length >= 3 || family === 't5';
      if (usable && token.startsWith(family) && family.length > (best?.length ?? 0)) best = family;
    }
    if (best) return best;
  }
  return undefined;
}

/**
 * First family match across candidate tokens. For Hugging Face models
 * the tags array contains the model_type token ("qwen2", "llama"...),
 * so tags are preferred, then the model id, then the raw family field.
 */
export function detectFamilyId(id: string, tags: string[], family?: string): string | undefined {
  for (const tag of tags) {
    const hit = normalizeFamily(tag);
    if (hit) return hit;
  }
  const fromId = normalizeFamily(id);
  if (fromId) return fromId;
  if (family) return normalizeFamily(family);
  return undefined;
}

/** Canonical display label for a family id ("qwen" -> "Qwen"). */
export function familyLabel(familyId: string): string {
  if (familyId === 'gpt') return 'GPT';
  if (familyId === 'mpt' || familyId === 't5' || familyId === 'yi' || familyId === 'glm') {
    return familyId.toUpperCase();
  }
  return familyId.charAt(0).toUpperCase() + familyId.slice(1);
}
