/*
 * ProsperoAI — diagnostics
 *
 * Structured log sink (whitepaper §33). Severity + subsystem prefixes,
 * forwarded to the kernel log on PS5 payload builds.
 */

#include <pai/log.h>

#include <stdarg.h>
#include <stdio.h>

#ifdef PAI_PS5
#include <ps5/klog.h>
#endif

static const char *const k_level_name[PAI_LOG_COUNT] = {
    [PAI_LOG_TRACE] = "TRACE",
    [PAI_LOG_DEBUG] = "DEBUG",
    [PAI_LOG_INFO]  = "INFO",
    [PAI_LOG_WARN]  = "WARN",
    [PAI_LOG_ERROR] = "ERROR",
};

static const char *const k_subsystem_name[PAI_SUB_COUNT] = {
    [PAI_SUB_CORE]     = "core",
    [PAI_SUB_PLATFORM] = "platform",
    [PAI_SUB_GPU]      = "gpu",
    [PAI_SUB_TENSOR]   = "tensor",
    [PAI_SUB_REF]      = "ref",
    [PAI_SUB_SCHED]    = "sched",
    [PAI_SUB_PROTO]    = "proto",
    [PAI_SUB_PROFILER] = "profiler",
    [PAI_SUB_DIAG]     = "diag",
};

static pai_log_level_t g_pai_log_level = PAI_LOG_INFO;

void
pai_log_set_level(pai_log_level_t level) {
  if ((uint32_t)level < PAI_LOG_COUNT) {
    g_pai_log_level = level;
  }
}

pai_log_level_t
pai_log_get_level(void) {
  return g_pai_log_level;
}

void
pai_log(pai_log_level_t level, pai_log_subsystem_t subsystem, const char *fmt,
        ...) {
  char buf[1024];
  va_list ap;
  int len;

  if ((uint32_t)level < g_pai_log_level ||
      (uint32_t)level >= PAI_LOG_COUNT ||
      (uint32_t)subsystem >= PAI_SUB_COUNT) {
    return;
  }

  len = snprintf(buf, sizeof(buf), "[%s|%s] ", k_level_name[level],
                 k_subsystem_name[subsystem]);
  if (len < 0 || (size_t)len >= sizeof(buf)) {
    return;
  }

  va_start(ap, fmt);
  vsnprintf(buf + len, sizeof(buf) - (size_t)len, fmt, ap);
  va_end(ap);

  printf("%s", buf);
#ifdef PAI_PS5
  klog_puts(buf);
#endif
}
