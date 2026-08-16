# ProsperoAI — optimized CPU backend (§6/§12)
#
# SIMD-accelerated operators with runtime dispatch (portable / SSE2 /
# AVX2+FMA). The AVX2 tier uses function-level target attributes, so no
# separate -march flags are needed.
pai_define_library(pai_cpu_optimized
  opt_ops.c
)
