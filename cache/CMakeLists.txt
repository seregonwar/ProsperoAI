# ProsperoAI — KV Cache Manager (whitepaper §19)
#
# Per-session transformer KV cache with buddy-backed block allocation
# (§16 GPU allocator foundation), prefix caching and scheduler
# integration hooks. Quantization / spill arrive in Phase 3+.
pai_define_library(pai_cache
  kv_cache.c
)

target_include_directories(pai_cache PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(pai_cache PUBLIC
  pai_memory
)
