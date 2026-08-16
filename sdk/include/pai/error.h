/*
 * ProsperoAI — public SDK
 *
 * Status codes shared by every PAI API. The high-level API (pai/api.h)
 * guarantees ABI stability; these codes are part of that contract.
 */

#ifndef PAI_ERROR_H
#define PAI_ERROR_H

#include <stdint.h>

typedef int32_t pai_status_t;

#define PAI_OK                0
#define PAI_ERR_INVALID_ARG  -1
#define PAI_ERR_NOMEM        -2
#define PAI_ERR_UNSUPPORTED  -3
#define PAI_ERR_CAPABILITY   -4
#define PAI_ERR_TIMEOUT      -5
#define PAI_ERR_MISMATCH     -6
#define PAI_ERR_IO           -7
#define PAI_ERR_INIT         -8
#define PAI_ERR_INTERNAL     -9
#define PAI_ERR_PROTOCOL    -10 /* wire protocol violation (whitepaper §24) */

const char *pai_status_str(pai_status_t status);

#endif /* PAI_ERROR_H */
