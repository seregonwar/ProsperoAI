set(PAI_TESTS
  test_tensor
  test_arena
  test_memory_planner
  test_allocator
  test_graph
  test_ir
  test_scheduler
  test_tokenizer
  test_sampler
  test_kv_cache
  test_pai
  test_model
  test_quantize
  test_pai_opt
  test_opt_ops
  test_importer
  test_gguf
  test_pm4
  test_ref_ops
  test_op_registry
  test_profiler
  test_hal_host
  test_compress
  test_protocol
  test_tcp
  test_json
  test_gateway
  test_fuzz
)

foreach(test_name ${PAI_TESTS})
  add_executable(${test_name} ${test_name}.c)
  target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${test_name} PRIVATE
    pai_runtime
    pai_tensor
    pai_memory
    pai_compression
    pai_graph
    pai_ir
    pai_scheduler
    pai_cache
    pai_pai
    pai_models
    pai_llama
    pai_protocol
    pai_pm4
    pai_cpu_reference
    pai_cpu_optimized
    pai_gpu_hal
    pai_gpu_kernels
    pai_profiler
    pai_diag
    pai_gateway
  )
  add_test(NAME ${test_name} COMMAND ${test_name})
  pai_enable_big_stack(${test_name})
endforeach()

# T7 exit-test harness (target defined in toolchain/CMakeLists.txt):
# MLP 8->16->8 differential - CPU-ref executor + eager registry + plan
# profile. The GPU/host-ref leg lands with T4C.
add_test(NAME mlp_test COMMAND mlp_test)

