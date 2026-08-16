/*
 * ProsperoAI — PS5 privilege escalation + jail escape
 *
 * Sequence mirrors MemDBG's memdbg_privilege_jailbreak_self (hardware-
 * proven on PS5 payloads): zero the credential ids, adopt the system
 * auth id with full caps, then retarget the process root/jail vnodes and
 * the filedesc rdir/jdir onto the real root vnode. Without this, the
 * payload is confined to the hijacked process' jail and cannot reach
 * /data.
 *
 * ProsperoAI stays escalated for the lifetime of the payload, so no
 * rollback is implemented; failures are reported per-step.
 */

#include "platform.h"

#include <pai/log.h>

#include <string.h>
#include <unistd.h>

#ifdef PAI_PS5

#include <ps5/kernel.h>

/* GPU-required auth id (OpenAGC agcProsperoPrepareGpuCredentials:
 * 0x4801000000000000). The MemDBG system auth id 0x...0013 is rejected
 * by the 9.40 GPU paths. */
#define PAI_SYSTEM_AUTHID 0x4801000000000000ULL

static const uint8_t k_pai_full_caps[16] = {
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
};

pai_status_t
pai_platform_escalate(void) {
  pid_t pid = getpid();
  intptr_t rootv;
  intptr_t fd;
  int failures = 0;

  failures += kernel_set_ucred_uid(pid, 0) != 0;
  failures += kernel_set_ucred_ruid(pid, 0) != 0;
  failures += kernel_set_ucred_svuid(pid, 0) != 0;
  failures += kernel_set_ucred_rgid(pid, 0) != 0;
  failures += kernel_set_ucred_svgid(pid, 0) != 0;
  failures += kernel_set_ucred_authid(pid, PAI_SYSTEM_AUTHID) != 0;
  failures += kernel_set_ucred_caps(pid, k_pai_full_caps) != 0;

  rootv = kernel_get_root_vnode();
  if (rootv == 0) {
    failures++;
    PAI_LOG_ERROR_(PAI_SUB_PLATFORM,
                   "escalate: root vnode unavailable\n");
    return PAI_ERR_CAPABILITY;
  }

  failures += kernel_set_proc_rootdir(pid, rootv) != 0;
  failures += kernel_set_proc_jaildir(pid, rootv) != 0;

  fd = kernel_get_proc_filedesc(pid);
  if (fd == 0) {
    failures++;
    PAI_LOG_ERROR_(PAI_SUB_PLATFORM,
                   "escalate: filedesc unavailable\n");
    return PAI_ERR_CAPABILITY;
  }

  failures +=
      kernel_setlong(fd + KERNEL_OFFSET_FILEDESC_FD_RDIR, (uint64_t)rootv) != 0;
  failures +=
      kernel_setlong(fd + KERNEL_OFFSET_FILEDESC_FD_JDIR, (uint64_t)rootv) != 0;

  if (failures != 0) {
    PAI_LOG_WARN_(PAI_SUB_PLATFORM,
                  "escalate: %d step(s) failed; filesystem access may be "
                  "limited\n",
                  failures);
    return PAI_ERR_CAPABILITY;
  }

  PAI_LOG_INFO_(PAI_SUB_PLATFORM,
                "escalate: payload escaped sandbox (root=0x%lx)\n",
                (unsigned long)rootv);
  return PAI_OK;
}

#endif /* PAI_PS5 */
