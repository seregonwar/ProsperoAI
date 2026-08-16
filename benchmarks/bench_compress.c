/*
 * ProsperoAI — compression benchmark (bench_compress)
 *
 * Measures encode/decode throughput and ratio of the clean-room
 * Kraken-family codec over model-realistic payloads. Usage:
 *   bench_compress [size_mb] [repeats]
 *
 * Scenarios:
 *   text    - token-like ASCII, periodic (best case for LZ)
 *   weights - f32 weight blocks with local redundancy
 *   mixed   - alternating compressible / random regions
 *   random  - incompressible (worst case, fallback paths)
 *   zeros   - maximum compressibility
 */

#include <bench.h>
#include <compression/pai_compress.h>

#include <pai/log.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t
now_ns(void) {
  return pai_bench_now_ns();
}

static void
fill_scenario(uint8_t *p, uint64_t n, const char *name) {
  uint64_t i;
  uint32_t x = 0x9E3779B1u;

  if (strcmp(name, "text") == 0) {
    const char *s =
        "the quick brown fox jumps over the lazy dog; layer %u weights "
        "and biases are stored k-major with group scales; ";
    uint64_t sl = strlen(s);
    for (i = 0; i < n; i++) {
      p[i] = (uint8_t)s[(i * 7u + (i >> 8) * 13u) % sl];
    }
  } else if (strcmp(name, "weights") == 0) {
    /* f32 blocks: exponent/mantissa structure + repeated blocks */
    for (i = 0; i < n; i++) {
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      p[i] = (uint8_t)(x >> 24);
      if ((i & 0xFFFu) >= 0xC00u) {
        p[i] = p[i - 0x400u];
      }
    }
  } else if (strcmp(name, "mixed") == 0) {
    for (i = 0; i < n; i++) {
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      if ((i >> 16) & 1u) {
        p[i] = (uint8_t)(x >> 24);
      } else {
        p[i] = (uint8_t)(i * 31u + (i >> 9));
      }
    }
  } else if (strcmp(name, "zeros") == 0) {
    memset(p, 0, (size_t)n);
  } else { /* random */
    for (i = 0; i < n; i++) {
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      p[i] = (uint8_t)(x >> 24);
    }
  }
}

static void
run_scenario(const char *name, uint64_t n, int repeats, double *best_enc,
             double *best_dec) {
  uint8_t *src = (uint8_t *)malloc((size_t)n);
  uint8_t *dec = (uint8_t *)malloc((size_t)n);
  uint64_t bound = pai_compress_bound(n) + 16u;
  uint8_t *comp = (uint8_t *)malloc((size_t)bound);
  uint64_t clen = 0, dlen = 0;
  double te = 0, td = 0;
  int r;

  if (src == NULL || dec == NULL || comp == NULL) {
    printf("  %-8s OOM\n", name);
    return;
  }
  fill_scenario(src, n, name);

  for (r = 0; r < repeats; r++) {
    uint64_t t0, t1;
    pai_status_t st;

    t0 = now_ns();
    st = pai_compress_encode(src, n, comp, bound, &clen);
    t1 = now_ns();
    if (st != PAI_OK) {
      printf("  %-8s encode failed (%d)\n", name, st);
      goto done;
    }
    te += (double)(t1 - t0) / (double)repeats;

    t0 = now_ns();
    st = pai_compress_decode(comp, clen, dec, n, &dlen);
    t1 = now_ns();
    if (st != PAI_OK || dlen != n || memcmp(src, dec, (size_t)n) != 0) {
      printf("  %-8s decode failed (%d)\n", name, st);
      goto done;
    }
    td += (double)(t1 - t0) / (double)repeats;
  }

  printf("  %-8s ratio %6.1f%%  encode %7.1f MB/s  decode %7.1f MB/s\n", name,
         100.0 * (double)clen / (double)n, (double)n / te / 1e6,
         (double)n / td / 1e6);
  if (best_enc != NULL && (double)n / te > *best_enc) {
    *best_enc = (double)n / te;
  }
  if (best_dec != NULL && (double)n / td > *best_dec) {
    *best_dec = (double)n / td;
  }

done:
  free(src);
  free(dec);
  free(comp);
}

int
main(int argc, char **argv) {
  uint64_t mb = argc > 1 ? (uint64_t)strtoull(argv[1], NULL, 10) : 16u;
  int repeats = argc > 2 ? atoi(argv[2]) : 3;
  uint64_t n = mb * 1024u * 1024u;
  double best_enc = 0, best_dec = 0;
  static const char *const scenarios[] = {"text", "weights", "mixed",
                                          "random", "zeros"};

  PAI_LOG_INFO_(PAI_SUB_CORE, "compression benchmark: %llu MiB x%d\n",
                (unsigned long long)mb, repeats);

  for (uint32_t i = 0; i < sizeof(scenarios) / sizeof(scenarios[0]); i++) {
    run_scenario(scenarios[i], n, repeats, &best_enc, &best_dec);
  }

  printf("  best encode %.1f MB/s, best decode %.1f MB/s\n", best_enc,
         best_dec);
  return 0;
}
