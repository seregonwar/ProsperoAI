/*
 * ProsperoAI — minimal JSON (gateway §26)
 *
 * A small, dependency-free JSON document model for the OpenAI-compatible
 * gateway: a DOM parser (node pool + string arena) and a writer
 * (growable buffer with proper escaping and number formatting). It is
 * deliberately strict (RFC 8259 shape) and bounded:
 *
 *   - maximum nesting depth PAI_JSON_MAX_DEPTH;
 *   - maximum node count PAI_JSON_MAX_NODES (DoS bound);
 *   - per-string length cap PAI_JSON_MAX_STRING;
 *
 * Objects store members as child (key, value) node pairs; keys are
 * copies held in the document's string arena. The parser rejects
 * trailing garbage, control characters in strings and malformed
 * numbers. \uXXXX escapes are decoded to UTF-8 including surrogate
 * pairs.
 *
 * The writer serializes the parsed tree back to text (round-trip for
 * tests) and provides the low-level pieces the gateway uses to build
 * responses: pai_json_wb_putf for formatting and pai_json_quote for
 * escaping user content.
 */

#ifndef PAI_GATEWAY_JSON_H
#define PAI_GATEWAY_JSON_H

#include <pai/error.h>

#include <stdarg.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PAI_JSON_MAX_DEPTH 32
#define PAI_JSON_MAX_NODES 8192
#define PAI_JSON_MAX_STRING (1u << 20)

typedef enum pai_json_type {
  PAI_JSON_NULL = 0,
  PAI_JSON_BOOL,
  PAI_JSON_NUMBER,
  PAI_JSON_STRING,
  PAI_JSON_ARRAY,
  PAI_JSON_OBJECT,
} pai_json_type_t;

/* A parsed document. Nodes and strings live in owned pools. */
typedef struct pai_json_doc {
  struct pai_json_node *nodes;
  uint32_t num_nodes;
  uint32_t cap_nodes;
  char *strs;             /* string arena (NUL-terminated entries)     */
  uint32_t str_used;
  uint32_t str_cap;
  int32_t root;           /* root node index; -1 when the input is
                             empty or whitespace-only                    */
} pai_json_doc_t;

/*
 * Parse `nbytes` of JSON text. PAI_ERR_PROTOCOL on malformed input,
 * PAI_ERR_NOMEM when a size limit is exceeded. The document must be
 * destroyed with pai_json_destroy.
 */
pai_status_t pai_json_parse(pai_json_doc_t *doc, const char *text,
                            uint32_t nbytes);
void pai_json_destroy(pai_json_doc_t *doc);

/* Root node index (-1 for an empty document). */
int32_t pai_json_root(const pai_json_doc_t *doc);

/* Node accessors. */
pai_json_type_t pai_json_type(const pai_json_doc_t *doc, int32_t node);
const char *pai_json_str(const pai_json_doc_t *doc, int32_t node); /* NULL when not a string */
double pai_json_num(const pai_json_doc_t *doc, int32_t node);      /* 0 when not a number */
int pai_json_bool(const pai_json_doc_t *doc, int32_t node);        /* 0 when not a bool */

/* Object member lookup: node index of the value, or -1. */
int32_t pai_json_member(const pai_json_doc_t *doc, int32_t obj,
                        const char *key);

/* Array access. */
int32_t pai_json_array_len(const pai_json_doc_t *doc, int32_t arr); /* -1 when not an array */
int32_t pai_json_array_at(const pai_json_doc_t *doc, int32_t arr,
                          uint32_t i);

/* Convenience: string member value or NULL when absent/not a string. */
const char *pai_json_str_member(const pai_json_doc_t *doc, int32_t obj,
                                const char *key);

/* ------------------------------------------------------------------ */
/* Writer                                                              */
/* ------------------------------------------------------------------ */

/* Growable byte buffer (heap owned). */
typedef struct pai_json_wb {
  char *buf;
  uint32_t len;
  uint32_t cap;
} pai_json_wb_t;

void pai_json_wb_init(pai_json_wb_t *wb);
void pai_json_wb_destroy(pai_json_wb_t *wb);

pai_status_t pai_json_wb_reserve(pai_json_wb_t *wb, uint32_t extra);
pai_status_t pai_json_wb_putc(pai_json_wb_t *wb, char c);
pai_status_t pai_json_wb_putn(pai_json_wb_t *wb, const char *s, uint32_t n);
pai_status_t pai_json_wb_puts(pai_json_wb_t *wb, const char *s);
pai_status_t pai_json_wb_putf(pai_json_wb_t *wb, const char *fmt, ...);

/* Append `s` (nbytes) as a quoted, escaped JSON string. */
pai_status_t pai_json_quote(pai_json_wb_t *wb, const char *s, uint32_t n);

/* Serialize the subtree at `node` back to JSON text. */
pai_status_t pai_json_serialize(const pai_json_doc_t *doc, int32_t node,
                                pai_json_wb_t *wb);

/* Append a number with a short, round-trippable format ("%.7g"). */
pai_status_t pai_json_put_number(pai_json_wb_t *wb, double v);

#ifdef __cplusplus
}
#endif

#endif /* PAI_GATEWAY_JSON_H */
