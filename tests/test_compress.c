#include "test.h"

#include <compression/pai_compress.h>
#include <pai/pai.h>

#include <stdlib.h>
#include <string.h>

static void
fill_pattern(uint8_t *p, uint64_t n, uint32_t seed) {
  uint32_t x = seed | 1u;
  uint64_t i;
  for (i = 0; i < n; i++) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    p[i] = (uint8_t)(x >> 24);
  }
}

static void
roundtrip(const uint8_t *src, uint64_t n) {
  uint64_t bound = pai_compress_bound(n);
  uint8_t *comp = (uint8_t *)malloc((size_t)bound + 64u);
  uint8_t *dec = (uint8_t *)malloc((size_t)n + 1u);
  uint64_t comp_len = 0, dec_len = 0;
  uint32_t method = 0;

  CHECK(comp != NULL && dec != NULL);
  CHECK(pai_compress_encode(src, n, comp, bound + 64u, &comp_len) == PAI_OK);
  CHECK(comp_len > 0 && comp_len <= bound + 64u);
  CHECK(pai_compress_probe(comp, comp_len, &method) == PAI_OK);
  CHECK_EQ_UINT(method, PAI_COMP_METHOD_KRAKEN);
  CHECK(pai_compress_decode(comp, comp_len, dec, n + 1u, &dec_len) == PAI_OK);
  CHECK_EQ_UINT(dec_len, n);
  CHECK(memcmp(src, dec, (size_t)n) == 0);

  free(comp);
  free(dec);
}

TEST_MAIN_BEGIN()

{
  uint8_t buf[0x10000];
  uint64_t sizes[] = {1, 2, 7, 8, 9, 15, 16, 63, 64, 65, 255, 256, 1000,
                      4096, 0xFFFF, 0x10000};
  uint64_t i;

  /* random data (mostly incompressible) */
  for (i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
    uint8_t *p = (uint8_t *)malloc((size_t)sizes[i]);
    fill_pattern(p, sizes[i], (uint32_t)i * 0x9E37u + 1u);
    roundtrip(p, sizes[i]);
    free(p);
  }

  /* zeros (maximally compressible) */
  memset(buf, 0, sizeof(buf));
  roundtrip(buf, sizeof(buf));

  /* repeated pattern with long matches */
  fill_pattern(buf, 4096, 7u);
  for (i = 4096; i < sizeof(buf); i++) {
    buf[i] = buf[i - 4096];
  }
  roundtrip(buf, sizeof(buf));

  /* text-like (short matches) */
  for (i = 0; i < sizeof(buf); i++) {
    buf[i] = (uint8_t)('a' + (i % 26));
  }
  roundtrip(buf, sizeof(buf));

  /* whole quantum boundary crossing */
  {
    uint8_t *big = (uint8_t *)malloc(0x50000u);
    fill_pattern(big, 0x50000u, 0x12345u);
    roundtrip(big, 0x50000u);
    free(big);
  }

  /* probe: raw Kraken-family header */
  {
    uint8_t raw[4] = {0x0C, 10u, 0, 0};
    uint32_t method = 0;
    CHECK(pai_compress_probe(raw, 2, &method) == PAI_OK);
    CHECK_EQ_UINT(method, PAI_COMP_METHOD_AMPR);
  }
  {
    uint8_t none[4] = {1, 2, 3, 4};
    uint32_t method = 0;
    CHECK(pai_compress_probe(none, 4, &method) == PAI_OK);
    CHECK_EQ_UINT(method, PAI_COMP_METHOD_NONE);
  }

  /* garbage must fail cleanly */
  {
    uint8_t g[64];
    uint8_t out[64];
    uint64_t out_len = 0;
    fill_pattern(g, sizeof(g), 0xDEADu);
    CHECK(pai_compress_decode(g, sizeof(g), out, sizeof(out), &out_len) ==
          PAI_ERR_UNSUPPORTED);
  }

  /* truncated envelope must fail cleanly */
  {
    uint8_t *comp = (uint8_t *)malloc(pai_compress_bound(1000) + 64u);
    uint8_t dec[256];
    uint64_t clen = 0, dlen = 0;
    CHECK(pai_compress_encode(buf, 1000, comp,
                              pai_compress_bound(1000) + 64u, &clen) ==
          PAI_OK);
    CHECK(pai_compress_decode(comp, clen - 3u, dec, sizeof(dec), &dlen) !=
          PAI_OK);
    free(comp);
  }

  /* undersized destination */
  {
    uint8_t *comp = (uint8_t *)malloc(pai_compress_bound(1000) + 64u);
    uint8_t dec[64];
    uint64_t clen = 0, dlen = 0;
    CHECK(pai_compress_encode(buf, 1000, comp,
                              pai_compress_bound(1000) + 64u, &clen) ==
          PAI_OK);
    CHECK(pai_compress_decode(comp, clen, dec, sizeof(dec), &dlen) != PAI_OK);
    free(comp);
  }
  /* container plumbing: a compressed WEIGHTS section round-trips
   * through the builder + reader */
  {
    uint8_t weights[4096];
    uint8_t *comp =
        (uint8_t *)malloc(pai_compress_bound(sizeof(weights)) + 16u);
    uint64_t comp_n = 0;
    uint8_t *blob = (uint8_t *)malloc(0x10000u);
    uint32_t blob_n = 0;
    pai_pai_builder_t b;
    pai_pai_container_t c;
    const uint8_t *sec;
    uint32_t sec_n = 0;
    uint64_t dec_n = 0;
    uint8_t *dec = (uint8_t *)malloc(sizeof(weights));

    fill_pattern(weights, sizeof(weights), 0x77u);
    CHECK(pai_compress_encode(weights, sizeof(weights), comp,
                              pai_compress_bound(sizeof(weights)) + 16u,
                              &comp_n) == PAI_OK);

    pai_pai_builder_init(&b);
    CHECK(pai_pai_builder_add_flags(&b, PAI_PAI_SEC_WEIGHTS,
                                    PAI_PAI_SEC_FLAG_COMPRESSED, comp,
                                    (uint32_t)comp_n) == PAI_OK);
    CHECK(pai_pai_build(&b, blob, 0x10000u, &blob_n) == PAI_OK);
    CHECK(pai_pai_open(blob, blob_n, &c) == PAI_OK);
    CHECK(pai_pai_section_flags(&c, PAI_PAI_SEC_WEIGHTS) &
          PAI_PAI_SEC_FLAG_COMPRESSED);
    sec = pai_pai_section(&c, PAI_PAI_SEC_WEIGHTS, &sec_n);
    CHECK(sec != NULL && sec_n == comp_n);
    CHECK(pai_compress_decode(sec, sec_n, dec, sizeof(weights), &dec_n) ==
          PAI_OK);
    CHECK_EQ_UINT(dec_n, sizeof(weights));
    CHECK(memcmp(dec, weights, sizeof(weights)) == 0);

    free(dec);
    free(blob);
    free(comp);
  }

  /* flags on a non-WEIGHTS section must be rejected by the reader */
  {
    uint8_t blob[512];
    uint32_t blob_n = 0;
    pai_pai_builder_t b;
    pai_pai_container_t c;
    uint8_t payload[4] = {1, 2, 3, 4};

    pai_pai_builder_init(&b);
    pai_pai_builder_add_flags(&b, PAI_PAI_SEC_META,
                              PAI_PAI_SEC_FLAG_COMPRESSED, payload,
                              sizeof(payload));
    CHECK(pai_pai_build(&b, blob, sizeof(blob), &blob_n) == PAI_OK);
    CHECK(pai_pai_open(blob, blob_n, &c) == PAI_ERR_PROTOCOL);
  }
}

TEST_MAIN_END()
