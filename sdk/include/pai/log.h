/*
 * ProsperoAI — public SDK
 *
 * Structured logging with severity levels and subsystem identifiers
 * (whitepaper §33). On PS5 payload builds, messages are additionally
 * forwarded to the kernel log.
 */

#ifndef PAI_LOG_H
#define PAI_LOG_H

#include <stdint.h>

typedef enum pai_log_level {
  PAI_LOG_TRACE = 0,
  PAI_LOG_DEBUG,
  PAI_LOG_INFO,
  PAI_LOG_WARN,
  PAI_LOG_ERROR,
  PAI_LOG_COUNT
} pai_log_level_t;

typedef enum pai_log_subsystem {
  PAI_SUB_CORE = 0,
  PAI_SUB_PLATFORM,
  PAI_SUB_GPU,
  PAI_SUB_TENSOR,
  PAI_SUB_REF,
  PAI_SUB_SCHED,
  PAI_SUB_PROTO,
  PAI_SUB_PROFILER,
  PAI_SUB_DIAG,
  PAI_SUB_COUNT
} pai_log_subsystem_t;

void pai_log_set_level(pai_log_level_t level);
pai_log_level_t pai_log_get_level(void);

/*
 * Attach a file sink: every subsequent log line is appended there too.
 * Returns 0 on success. `path` must already be creatable (parent dirs
 * must exist).
 */
int pai_log_file_open(const char *path);
void pai_log_file_close(void);

void pai_log(pai_log_level_t level, pai_log_subsystem_t subsystem,
             const char *fmt, ...);

#define PAI_LOG_TRACE_(sub, ...) pai_log(PAI_LOG_TRACE, sub, __VA_ARGS__)
#define PAI_LOG_DEBUG_(sub, ...) pai_log(PAI_LOG_DEBUG, sub, __VA_ARGS__)
#define PAI_LOG_INFO_(sub, ...)  pai_log(PAI_LOG_INFO, sub, __VA_ARGS__)
#define PAI_LOG_WARN_(sub, ...)  pai_log(PAI_LOG_WARN, sub, __VA_ARGS__)
#define PAI_LOG_ERROR_(sub, ...) pai_log(PAI_LOG_ERROR, sub, __VA_ARGS__)

#endif /* PAI_LOG_H */
