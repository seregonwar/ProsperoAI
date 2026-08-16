# ProsperoAI — `.pai` Model Container (whitepaper §20)
#
# The native ProsperoAI model package: a portable container (metadata,
# tensor manifest, canonical weights, Prospero IR, tokenizer) with
# whole-file CRC integrity and reserved hardware-slice sections.
pai_define_library(pai_pai
  pai.c
)

target_include_directories(pai_pai PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(pai_pai PUBLIC
  pai_tensor
  pai_protocol
)
