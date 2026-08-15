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
  set(PAI_DEPLOY_ARGS "")
  if(DEFINED ENV{PS5_HOST})
    list(APPEND PAI_DEPLOY_ARGS -h $ENV{PS5_HOST})
  endif()
  if(DEFINED ENV{PS5_PORT})
    list(APPEND PAI_DEPLOY_ARGS -p $ENV{PS5_PORT})
  endif()

  if(CMAKE_HOST_WIN32)
    set(PAI_DEPLOY_CMD "${PS5_PAYLOAD_SDK}/win/prospero-deploy.cmd")
  else()
    set(PAI_DEPLOY_CMD "${PS5_DEPLOY}")
  endif()

  add_custom_target(pai-deploy
    USES_TERMINAL
    COMMAND ${PAI_DEPLOY_CMD} ${PAI_DEPLOY_ARGS} $<TARGET_FILE:${PAI_PAYLOAD_NAME}>
    DEPENDS ${PAI_PAYLOAD_NAME}
  )
endif()
