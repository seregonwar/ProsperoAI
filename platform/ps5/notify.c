/*
 * ProsperoAI — PS5 on-console notification
 *
 * Shows a deployment banner through sceKernelSendNotificationRequest
 * (layout from the PS5-Firmware-Spoofer, hardware-proven).
 */

#include "platform.h"

#include <pai/log.h>

#include <stdio.h>
#include <string.h>

#ifdef PAI_PS5

struct pai_notify_request {
  char useless1[45];
  char message[1024];
  char useless2[2051];
};

extern int sceKernelSendNotificationRequest(size_t,
                                            const struct pai_notify_request *,
                                            size_t, int);

void
pai_notify(const char *message) {
  struct pai_notify_request req;
  size_t len;

  memset(&req, 0, sizeof(req));

  len = strlen(message);
  if (len >= sizeof(req.message)) {
    len = sizeof(req.message) - 1;
  }
  memcpy(req.message, message, len);
  while (len > 0 && req.message[len - 1] == '\n') {
    req.message[--len] = '\0';
  }

  sceKernelSendNotificationRequest(0, &req, sizeof(req), 0);
  PAI_LOG_INFO_(PAI_SUB_PLATFORM, "notify: %s\n", req.message);
}

#endif /* PAI_PS5 */
