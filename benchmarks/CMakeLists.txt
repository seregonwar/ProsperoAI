# ProsperoAI — benchmark harness (§31)
pai_define_library(pai_bench
  bench.c
)

if(PAI_PLATFORM STREQUAL "host")
  add_executable(bench_gemm bench_gemm.c)
  target_link_libraries(bench_gemm PRIVATE
    pai_bench
    pai_cpu_optimized
    pai_cpu_reference
  )
  pai_enable_big_stack(bench_gemm)

  add_executable(bench_compress bench_compress.c)
  target_link_libraries(bench_compress PRIVATE
    pai_bench
    pai_compression
  )
endif()
