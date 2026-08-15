/*
 * ProsperoAI — PS5 platform detection
 *
 * Probes the payload environment without assuming any firmware layout:
 *   - firmware version via the payload host's kernel interface
 *   - direct-memory syscalls (probe alloc/map/free)
 *   - /dev/gc (probe open/close)
 */

#include "platform.h"

#include <pai/log.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef PAI_PS5

#include <ps5/kernel.h>

extern int sceKernelGetHwModelName(char *);
extern int sceKernelAllocateMainDirectMemory(size_t len, size_t alignment,
                                             int type, off_t *out);
extern int sceKernelMapNamedDirectMemory(void **addr, size_t len, int prot,
                                         int flags, off_t phys, size_t align,
                                         const char *name);

#define PROT_GPU_READ  0x10
#define PROT_GPU_WRITE 0x20
#define MAP_NO_COALESCE 0x400000

static void
pai_fnv1a32_update(uint32_t *hash, const void *data, size_t len) {
  const uint8_t *p = (const uint8_t *)data;
  for (size_t i = 0; i < len; i++) {
    *hash ^= p[i];
    *hash *= 0x01000193u;
  }
}

static uint32_t
pai_probe_dmem(void) {
  off_t phys = 0;
  void *va = NULL;
  int r;

  r = sceKernelAllocateMainDirectMemory(PAI_GPU_ALLOC_ALIGN, PAI_GPU_ALLOC_ALIGN,
                                        1, &phys);
  if (r != 0) {
    return 0;
  }

  r = sceKernelMapNamedDirectMemory(&va, PAI_GPU_ALLOC_ALIGN,
                                    PROT_READ | PROT_WRITE | PROT_GPU_READ |
                                        PROT_GPU_WRITE,
                                    MAP_NO_COALESCE, phys, PAI_GPU_ALLOC_ALIGN,
                                    "pai-probe");
  if (r != 0) {
    return 0;
  }

  munmap(va, PAI_GPU_ALLOC_ALIGN);
  return PAI_CAP_DMEM;
}

static uint32_t
pai_probe_dev_gc(void) {
  int fd = open("/dev/gc", O_RDWR);
  if (fd < 0) {
    return 0;
  }
  close(fd);
  return PAI_CAP_DEV_GC;
}

pai_status_t
pai_platform_detect(pai_platform_info_t *info) {
  uint32_t hash = 0x811c9dc5u;
  char fw_str[16];

  memset(info, 0, sizeof(*info));

  info->fw_version = (uint32_t)kernel_get_fw_version();

  if (sceKernelGetHwModelName(info->model) != 0 || info->model[0] == '\0') {
    snprintf(info->model, sizeof(info->model), "unknown");
  }

  if (info->fw_version != 0) {
    info->caps |= PAI_CAP_KERNEL_RW;
  }
  info->caps |= pai_probe_dmem();
  info->caps |= pai_probe_dev_gc();

#ifdef PAI_SAFE_MODE
  info->caps |= PAI_CAP_SAFE_MODE;
#endif

  snprintf(fw_str, sizeof(fw_str), "%08x", info->fw_version);
  pai_fnv1a32_update(&hash, fw_str, strlen(fw_str));
  pai_fnv1a32_update(&hash, info->model, strlen(info->model));
  snprintf(info->fingerprint, sizeof(info->fingerprint), "%08x", hash);

  return PAI_OK;
}

pai_gpu_backend_t
pai_platform_default_gpu_backend(const pai_platform_info_t *info) {
#ifdef PAI_SAFE_MODE
  return PAI_GPU_BACKEND_NONE;
#else
  if ((info->caps & (PAI_CAP_DMEM | PAI_CAP_DEV_GC)) ==
      (PAI_CAP_DMEM | PAI_CAP_DEV_GC)) {
    return PAI_GPU_BACKEND_PS5_GC;
  }
  return PAI_GPU_BACKEND_NONE;
#endif
}

#endif /* PAI_PS5 */
