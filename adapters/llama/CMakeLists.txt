# ProsperoAI — llama.cpp/GGUF adapter (whitepaper §9.3)
#
# GGUF reader + LLaMA-architecture translator. Host-side tooling only:
# it runs on the desktop, never on the console payload.
if(PAI_PLATFORM STREQUAL "host")
  pai_define_library(pai_llama
    gguf.c
    llama.c
  )
  target_link_libraries(pai_llama PUBLIC
    pai_models
    pai_pai
    pai_ir
    pai_graph
    pai_cpu_reference
    pai_tensor
  )
endif()
