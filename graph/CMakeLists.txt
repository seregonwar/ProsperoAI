pai_define_library(pai_graph
  graph.c
)

target_link_libraries(pai_graph PUBLIC
  pai_tensor
  pai_memory
)
