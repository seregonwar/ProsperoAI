/* Minimal test harness for PAI host tests. */

#ifndef PAI_TEST_H
#define PAI_TEST_H

#include <stdio.h>

static int g_pai_test_failures;

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);               \
      g_pai_test_failures++;                                                 \
    }                                                                        \
  } while (0)

#define CHECK_EQ_INT(a, b)                                                   \
  do {                                                                       \
    long long va_ = (long long)(a);                                          \
    long long vb_ = (long long)(b);                                          \
    if (va_ != vb_) {                                                        \
      printf("  FAIL %s:%d: %s == %s (%lld != %lld)\n", __FILE__, __LINE__,  \
             #a, #b, va_, vb_);                                              \
      g_pai_test_failures++;                                                 \
    }                                                                        \
  } while (0)

#define CHECK_EQ_UINT(a, b)                                                  \
  do {                                                                       \
    unsigned long long va_ = (unsigned long long)(a);                        \
    unsigned long long vb_ = (unsigned long long)(b);                        \
    if (va_ != vb_) {                                                        \
      printf("  FAIL %s:%d: %s == %s (%llu != %llu)\n", __FILE__, __LINE__,  \
             #a, #b, va_, vb_);                                              \
      g_pai_test_failures++;                                                 \
    }                                                                        \
  } while (0)

#define TEST_MAIN_BEGIN()                                                    \
  int main(void) {                                                           \
    g_pai_test_failures = 0;

#define TEST_MAIN_END()                                                      \
    if (g_pai_test_failures) {                                               \
      printf("FAILED (%d)\n", g_pai_test_failures);                          \
      return 1;                                                              \
    }                                                                        \
    printf("ok\n");                                                          \
    return 0;                                                                \
  }

#endif /* PAI_TEST_H */
