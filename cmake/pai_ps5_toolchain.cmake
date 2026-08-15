# ProsperoAI — PS5 cross toolchain
#
# Wraps the OpenOrbis ps5-payload-sdk prospero.cmake toolchain and sets
# PAI-specific platform state.

set(PAI_PS5_SDK "${CMAKE_CURRENT_LIST_DIR}/../ps5-payload-sdk" CACHE PATH
    "Path to the ps5-payload-sdk checkout")

if(NOT EXISTS "${PAI_PS5_SDK}/toolchain/prospero.cmake")
  message(FATAL_ERROR
    "ps5-payload-sdk not found at ${PAI_PS5_SDK}. "
    "Expected ${PAI_PS5_SDK}/toolchain/prospero.cmake")
endif()

include("${PAI_PS5_SDK}/toolchain/prospero.cmake")

set(PAI_PLATFORM "ps5" CACHE STRING "Target platform" FORCE)

# The PS5 is a Zen 2 x86-64 machine; keep CPU intrinsics explicit at
# compile time instead of relying on the host compiler's defaults.
add_compile_definitions(PAI_PS5=1 PAI_ARCH_X86_64=1)
