#include <pai/error.h>

#include <pai/version.h>

#include <stdio.h>

const char *
pai_status_str(pai_status_t status) {
  switch (status) {
  case PAI_OK:
    return "ok";
  case PAI_ERR_INVALID_ARG:
    return "invalid argument";
  case PAI_ERR_NOMEM:
    return "out of memory";
  case PAI_ERR_UNSUPPORTED:
    return "unsupported";
  case PAI_ERR_CAPABILITY:
    return "capability unavailable";
  case PAI_ERR_TIMEOUT:
    return "timeout";
  case PAI_ERR_MISMATCH:
    return "validation mismatch";
  case PAI_ERR_IO:
    return "i/o error";
  case PAI_ERR_INIT:
    return "initialization failed";
  case PAI_ERR_INTERNAL:
    return "internal error";
  case PAI_ERR_PROTOCOL:
    return "protocol violation";
  default:
    return "unknown";
  }
}

const char *
pai_version_string(void) {
  static char buf[64];
  snprintf(buf, sizeof(buf), "v%d.%d.%d-%s", PAI_VERSION_MAJOR,
           PAI_VERSION_MINOR, PAI_VERSION_PATCH, PAI_MILESTONE);
  return buf;
}
