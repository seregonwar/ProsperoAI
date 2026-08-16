#include "m0_experiments.h"

const uint32_t pai_store_const_code[PAI_STORE_CONST_CODE_WORDS] = {
#include "gfx1013/pai_store_const.inc"
};

const uint32_t pai_store_const64_code[PAI_STORE_CONST64_CODE_WORDS] = {
#include "gfx1013/pai_store_const64.inc"
};

const uint32_t pai_store64_x2_code[PAI_STORE64_X2_CODE_WORDS] = {
#include "gfx1013/pai_store64_x2.inc"
};

const uint32_t pai_store64_x4_code[PAI_STORE64_X4_CODE_WORDS] = {
#include "gfx1013/pai_store64_x4.inc"
};

const uint32_t pai_loadstore_code[PAI_LOADSTORE_CODE_WORDS] = {
#include "gfx1013/pai_loadstore.inc"
};

const uint32_t pai_bisect_code[] = {
#include "gfx1013/pai_bisect.inc"
};
