# ProsperoAI — weight compression (clean-room Kraken-family codec)
pai_define_library(pai_compression
  pai_compress.c
  kraken_decode.c
  kraken_encode.c
)

target_include_directories(pai_compression PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(pai_compression PUBLIC pai_diag)
