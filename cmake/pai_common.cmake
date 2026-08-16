# ProsperoAI — shared CMake configuration
#
# Options, compile flags and helpers shared by every build configuration
# (PS5 cross builds and host builds).

option(PAI_BUILD_TESTS "Build unit tests (host builds only)" OFF)
option(PAI_SAFE_MODE "Safe mode: disable experimental GPU paths and optional kernel interaction" OFF)
option(PAI_ASSEMBLE_SHADERS "Assemble gfx1013 shaders at build time (requires llvm-mc)" ON)

set(PAI_PLATFORM "host" CACHE STRING "Target platform: host or ps5")
set_property(CACHE PAI_PLATFORM PROPERTY STRINGS host ps5)

set(PAI_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(PAI_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(PAI_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(PAI_MILESTONE "PAI-M0")

# C11 is the baseline dialect of the runtime.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /wd5105)
else()
  add_compile_options(-Wall -Wextra)
endif()

if(PAI_PLATFORM STREQUAL "ps5")
  add_compile_definitions(PAI_PS5=1)
else()
  add_compile_definitions(PAI_HOST=1)
endif()

if(PAI_SAFE_MODE)
  add_compile_definitions(PAI_SAFE_MODE=1)
  message(STATUS "ProsperoAI: SAFE MODE enabled (GPU paths disabled)")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(PAI_DEBUG=1)
endif()

# Every PAI static library uses these baseline properties.
function(pai_define_library target)
  add_library(${target} STATIC ${ARGN})
  target_include_directories(${target} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/sdk/include
  )
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

# Host executables can hold large fixed-capacity graph/IR structs on the
# stack (1024-value caps make pai_graph_t ~232 KB; debug builds give
# every local its own slot, so a few locals can exceed the default 1 MB
# reserve). Raise the reserved stack for such targets. Handles both the
# clang-cl (MSVC) and clang-GNU-driver + lld-link host toolchains.
function(pai_enable_big_stack target)
  if(MSVC)
    target_link_options(${target} PRIVATE /STACK:8388608)
  elseif(WIN32)
    target_link_options(${target} PRIVATE "-Wl,/STACK:8388608")
  endif()
endfunction()
