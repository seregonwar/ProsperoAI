# ProsperoAI — Prospero IR (whitepaper §10)
#
# Graph-level intermediate representation with flat binary serialization
# (the portable in-memory form embedded in .pai containers and carried
# over the Prospero Protocol). The graph <-> IR bridge (ir_graph.c)
# links against the graph layer; the core stays standalone.
pai_define_library(pai_ir
  ir.c
  ir_graph.c
  kir.c
)

target_include_directories(pai_ir PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(pai_ir PUBLIC
  pai_graph
  pai_tensor
  pai_protocol
)
