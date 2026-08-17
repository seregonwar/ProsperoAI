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
    pai_gpu_hal
    pai_tensor
    pai_diag
  )
  pai_enable_big_stack(mlp_test)

  # Phase 2 differential: on-GPU RoPE table generator ABI (G67/G68
  # ropegen mirror) vs the pai_ref_rope_cossin_f32 oracle.
  add_executable(ropegen_diff ropegen_diff.c)
  target_link_libraries(ropegen_diff PRIVATE
    pai_cpu_reference
    pai_gpu_hal
    pai_diag
  )
  pai_enable_big_stack(ropegen_diff)

  # Phase 2 contract: nonlinear serial decomposition (RMSNorm / SiLU /
  # softmax / attention row) vs the double oracle — pre-flight for the
  # serial GPU kernels (v_rsqrt / v_exp probes).
  add_executable(nonlinear_contract nonlinear_contract.c)
  target_link_libraries(nonlinear_contract PRIVATE
    pai_cpu_reference
    pai_diag
  )
  pai_enable_big_stack(nonlinear_contract)

  # Phase 2 reference path: small decoder forward pass (embed -> RoPE
  # -> causal GQA attention -> RMSNorm -> SiLU MLP -> logits -> greedy
  # first token) chained through ref_ops vs a double oracle.
  add_executable(decoder_test decoder_test.c)
  target_link_libraries(decoder_test PRIVATE
    pai_cpu_reference
    pai_gpu_hal
    pai_models
    pai_tensor
    pai_cache
    pai_diag
  )
  pai_enable_big_stack(decoder_test)
endif()
