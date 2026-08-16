# ProsperoAI — runtime compatibility layers (whitepaper §9)
#
# External-runtime adapters translate third-party model formats into
# native ProsperoAI containers. `llama/` is the first compatibility
# target (§9.3): GGUF model files -> native `.pai` (host tooling only).
add_subdirectory(llama)
