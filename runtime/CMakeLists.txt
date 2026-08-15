pai_define_library(pai_runtime
  runtime.c
  version.c
  api_stubs.c
)

target_link_libraries(pai_runtime PUBLIC
  pai_platform
  pai_memory
  pai_diag
)
