# ProsperoAI — `pai` command-line tooling (whitepaper §20)
#
# Host-side model tooling: convert / inspect / validate / benchmark.
if(PAI_PLATFORM STREQUAL "host")
  add_executable(pai pai_cli.c)
  target_link_libraries(pai PRIVATE
    pai_bench
    pai_models
    pai_llama
    pai_runtime
    pai_pai
    pai_ir
    pai_graph
    pai_scheduler
    pai_cache
    pai_cpu_optimized
    pai_cpu_reference
    pai_protocol
    pai_gateway
    pai_tensor
    pai_diag
  )
  pai_enable_big_stack(pai)

  # T7 exit test: MLP 8->16->8 differential harness (B leg: CPU-ref +
  # profile; A leg: GPU serial + host-ref plugs into mlp_test.c).
  add_executable(mlp_test mlp_test.c)
  target_link_libraries(mlp_test PRIVATE
    pai_runtime
    pai_graph
    pai_scheduler
    pai_profiler
    pai_cpu_reference
    pai_tensor
    pai_diag
  )
  pai_enable_big_stack(mlp_test)
endif()
