cmake_minimum_required(VERSION 3.20)

project(ProsperoAI
  VERSION 0.1.0
  DESCRIPTION "High-Performance General-Purpose AI Runtime for Jailbroken PlayStation 5"
  HOMEPAGE_URL ""
  LANGUAGES C
)

include(cmake/pai_common.cmake)

add_subdirectory(tensor)
add_subdirectory(memory)
add_subdirectory(compression)
add_subdirectory(graph)
add_subdirectory(ir)
add_subdirectory(scheduler)
add_subdirectory(cache)
add_subdirectory(pai)
add_subdirectory(models)
add_subdirectory(adapters)
add_subdirectory(protocol)
add_subdirectory(diagnostics)
add_subdirectory(profiler)
add_subdirectory(cpu/reference)
add_subdirectory(cpu/optimized)
add_subdirectory(benchmarks)
add_subdirectory(toolchain)
if(PAI_PLATFORM STREQUAL "host")
  add_subdirectory(gateway)
endif()
add_subdirectory(gpu/pm4)
add_subdirectory(gpu/hal)
add_subdirectory(gpu/kernels)
add_subdirectory(platform)
add_subdirectory(runtime)
add_subdirectory(payload)

if(PAI_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
