pai_define_library(pai_profiler
  bench.c
  profile.c
)

target_link_libraries(pai_profiler PUBLIC
  pai_scheduler
)
