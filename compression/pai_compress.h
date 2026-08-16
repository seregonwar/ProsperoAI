/*
 * ProsperoAI — weight compression (clean-room Kraken-family codec)
 *
 * Original implementation of the Oodle-Kraken-family container and the
 * Mermaid/Selkie LZ format, written from the published format behavior
 * (the format layout itself, re-derived from stream observations in the
 * open-source decompressors by Powzix/ooz and drakmor/ampr_emu, used as
 * documentation only — no code was copied).
 *
 * The AMPR method on PS5 is the same Oodle-Kraken family; both decode
 * through the same path here. Supported today:
 *   - Kraken container framing (2-byte header, per-quantum headers,
 *     raw/memset quanta, checksummed streams are accepted but the
 *     checksum is not verified)
 *   - Mermaid / Selkie quanta (same on-disk LZ format)
 *   - nested chunks: raw copy + RLE
 *   - our encoder emits Mermaid-format quanta with raw nested chunks
 * Unsupported (v2): LZNA, Leviathan, Bitknit, Huffman/TANS chunks.
 */

#ifndef PAI_COMPRESSION_H
#define PAI_COMPRESSION_H

#include <pai/error.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compression methods stored in containers. */
#define PAI_COMP_METHOD_NONE   0u
#define PAI_COMP_METHOD_KRAKEN 1u /* Kraken container (Mermaid/Selkie)    */
#define PAI_COMP_METHOD_AMPR   2u /* AMPR blob (Kraken-family, same code) */

/* Envelope written by pai_compress_encode: magic + method + dec size. */
#define PAI_COMP_MAGIC 0x43414B50u /* "PAKC" */

/* Upper bound for pai_compress_encode output. */
uint64_t pai_compress_bound(uint64_t len);

/*
 * Compress `len` bytes into `dst` (cap `cap`). Writes the self-describing
 * envelope (magic + method + decoded size) followed by a Kraken-family
 * stream. Worst case output is pai_compress_bound(len).
 */
pai_status_t pai_compress_encode(const uint8_t *src, uint64_t len,
                                 uint8_t *dst, uint64_t cap,
                                 uint64_t *out_len);

/*
 * Decompress into `dst` (cap `cap`). Accepts either an envelope produced
 * by pai_compress_encode or a raw Kraken-container stream (the latter is
 * decoded until `src` is exhausted). *out_len = decoded bytes.
 */
pai_status_t pai_compress_decode(const uint8_t *src, uint64_t len,
                                 uint8_t *dst, uint64_t cap,
                                 uint64_t *out_len);

/* Detect the method of a compressed blob (envelope or raw stream). */
pai_status_t pai_compress_probe(const uint8_t *src, uint64_t len,
                                uint32_t *out_method);

#ifdef __cplusplus
}
#endif

#endif /* PAI_COMPRESSION_H */
