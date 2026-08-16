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
endif()
