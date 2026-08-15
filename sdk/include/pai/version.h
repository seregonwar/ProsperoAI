/*
 * ProsperoAI — public SDK
 *
 * Runtime versioning. Keep PAI_VERSION_* in sync with the CMake project
 * version (cmake/pai_common.cmake).
 */

#ifndef PAI_VERSION_H
#define PAI_VERSION_H

#define PAI_VERSION_MAJOR 0
#define PAI_VERSION_MINOR 1
#define PAI_VERSION_PATCH 0

/* Development milestone of this build (whitepaper §40). */
#define PAI_MILESTONE "PAI-M0"

/* Human-readable runtime version string. */
const char *pai_version_string(void);

#endif /* PAI_VERSION_H */
