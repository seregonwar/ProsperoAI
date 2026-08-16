/*
 * ProsperoAI — runtime core. Bootstraps the single-payload architecture
 * in order (whitepaper §5): platform detection -> capability manager ->
 * memory -> GPU device; scheduler, protocol and models attach later.
 */

#ifndef PAI_RUNTIME_H
#define PAI_RUNTIME_H

#include <pai/api.h>

#include <arena.h>
#include <hal/hal.h>
#include <platform.h>

#define PAI_RUNTIME_ARENA_SIZE (1u << 16)

struct pai_runtime {
  pai_arena_t arena;
  uint8_t arena_mem[PAI_RUNTIME_ARENA_SIZE];
  pai_platform_info_t platform;
  pai_gpu_device_t *gpu;
  uint32_t initialized;
};

const pai_platform_info_t *pai_runtime_platform(const pai_runtime_t *runtime);
pai_gpu_device_t *pai_runtime_gpu(const pai_runtime_t *runtime);
pai_arena_t *pai_runtime_arena(pai_runtime_t *runtime);

#endif /* PAI_RUNTIME_H */
