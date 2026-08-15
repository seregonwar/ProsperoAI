if(PAI_PLATFORM STREQUAL "ps5")
  set(PAI_PLATFORM_SOURCES ps5/platform_ps5.c)
else()
  set(PAI_PLATFORM_SOURCES host/platform_host.c)
endif()

pai_define_library(pai_platform
  platform_log.c
  ${PAI_PLATFORM_SOURCES}
)

target_link_libraries(pai_platform PUBLIC pai_gpu_hal)

if(PAI_PLATFORM STREQUAL "ps5")
  target_link_libraries(pai_platform PRIVATE kernel_sys kernel_web)
endif()
