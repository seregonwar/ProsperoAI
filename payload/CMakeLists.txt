if(PAI_PLATFORM STREQUAL "ps5")
  set(PAI_PAYLOAD_NAME prosperoai.elf)
else()
  set(PAI_PAYLOAD_NAME pai_m0)
endif()

add_executable(${PAI_PAYLOAD_NAME} main.c)

target_link_libraries(${PAI_PAYLOAD_NAME} PRIVATE
  pai_runtime
  pai_gpu_hal
  pai_gpu_kernels
  pai_pm4
  pai_cpu_reference
  pai_profiler
  pai_tensor
)

if(PAI_PLATFORM STREQUAL "ps5")
  target_link_libraries(${PAI_PAYLOAD_NAME} PRIVATE kernel_sys kernel_web)
endif()

if(PAI_PLATFORM STREQUAL "ps5")
  add_custom_target(pai-deploy
    USES_TERMINAL
    COMMAND ${PS5_DEPLOY} $<TARGET_FILE:${PAI_PAYLOAD_NAME}>
    DEPENDS ${PAI_PAYLOAD_NAME}
  )
endif()
