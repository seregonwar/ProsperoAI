# T5A op registry: standalone library (no runtime deps) so both the
# runtime and the scheduler can link it without a cycle.
pai_define_library(pai_ops
  op_registry.c
)

target_link_libraries(pai_ops PUBLIC
  pai_cpu_reference
)

pai_define_library(pai_runtime
  runtime.c
  version.c
  api_impl.c
)

target_link_libraries(pai_runtime PUBLIC
  pai_platform
  pai_memory
  pai_diag
  pai_models
  pai_ops
)
