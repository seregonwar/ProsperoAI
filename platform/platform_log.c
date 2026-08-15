#include "platform.h"

#include <pai/log.h>

#include <stdio.h>

void
pai_platform_log(const pai_platform_info_t *info) {
  PAI_LOG_INFO_(PAI_SUB_PLATFORM, "firmware: 0x%08x\n", info->fw_version);
  PAI_LOG_INFO_(PAI_SUB_PLATFORM, "model: %s\n", info->model);
  PAI_LOG_INFO_(PAI_SUB_PLATFORM, "caps: 0x%08x\n", info->caps);
  PAI_LOG_INFO_(PAI_SUB_PLATFORM, "  kernel-rw   : %s\n",
                (info->caps & PAI_CAP_KERNEL_RW) ? "yes" : "no");
  PAI_LOG_INFO_(PAI_SUB_PLATFORM, "  dmem        : %s\n",
                (info->caps & PAI_CAP_DMEM) ? "yes" : "no");
  PAI_LOG_INFO_(PAI_SUB_PLATFORM, "  /dev/gc     : %s\n",
                (info->caps & PAI_CAP_DEV_GC) ? "yes" : "no");
  PAI_LOG_INFO_(PAI_SUB_PLATFORM, "  gpu-compute : %s\n",
                (info->caps & PAI_CAP_GPU_COMPUTE) ? "yes" : "no");
  PAI_LOG_INFO_(PAI_SUB_PLATFORM, "  avx2        : %s\n",
                (info->caps & PAI_CAP_CPU_AVX2) ? "yes" : "no");
  PAI_LOG_INFO_(PAI_SUB_PLATFORM, "fingerprint: %s\n", info->fingerprint);
}
