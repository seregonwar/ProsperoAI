if(PAI_PLATFORM STREQUAL "ps5")
  set(PAI_HAL_PS5_SOURCES backend_ps5_gc.c)
else()
  set(PAI_HAL_PS5_SOURCES "")
endif()

pai_define_library(pai_gpu_hal
  hal.c
  backend_host_ref.c
  host_kernels.c
  ${PAI_HAL_PS5_SOURCES}
)

target_include_directories(pai_gpu_hal PUBLIC
  ${PROJECT_SOURCE_DIR}/gpu
)
