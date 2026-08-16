# ProsperoAI — Scheduler (whitepaper §18) and execution planning (§11)
#
# Turns a compiled graph + static memory plan into an Execution Plan
# (device placement, per-step memory accounting, §17 memory modes) and
# ships the reference executor that runs the plan over the memory-plan
# region on host builds (Phase 1 vertical slice).
pai_define_library(pai_scheduler
  scheduler.c
)

target_include_directories(pai_scheduler PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(pai_scheduler PUBLIC
  pai_graph
  pai_memory
  pai_cpu_reference
  pai_ops
)
