set(PAI_TESTS
  test_tensor
  test_arena
  test_pm4
  test_ref_ops
  test_hal_host
  test_protocol
)

foreach(test_name ${PAI_TESTS})
  add_executable(${test_name} ${test_name}.c)
  target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${test_name} PRIVATE
    pai_runtime
    pai_tensor
    pai_memory
    pai_protocol
    pai_pm4
    pai_cpu_reference
    pai_gpu_hal
    pai_gpu_kernels
    pai_profiler
    pai_diag
  )
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
