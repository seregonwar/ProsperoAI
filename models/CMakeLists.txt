# ProsperoAI — model manager + CPU inference components
#
# Loads `.pai` containers into executable models, runs sessions over the
# reference executor, and ships the CPU-side tokenizer + sampler (§3.2).
pai_define_library(pai_models
  tokenizer.c
  sampler.c
  model.c
  quantize.c
  container.c
  importer.c
  pai_opt.c
)

target_include_directories(pai_models PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(pai_models PUBLIC
  pai_pai
  pai_ir
  pai_graph
  pai_scheduler
  pai_cache
  pai_tensor
  pai_compression
)
