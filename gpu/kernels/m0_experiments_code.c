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

const uint32_t pai_store64_smem_code[PAI_STORE64_SMEM_CODE_WORDS] = {
#include "gfx1013/pai_store64_smem.inc"
};

const uint32_t pai_store64_gold_code[PAI_STORE64_GOLD_CODE_WORDS] = {
#include "gfx1013/pai_store64_gold.inc"
};

const uint32_t pai_store64_v0_code[PAI_STORE64_V0_CODE_WORDS] = {
#include "gfx1013/pai_store64_v0.inc"
};

const uint32_t pai_mubufload_code[PAI_MUBUFLOAD_CODE_WORDS] = {
#include "gfx1013/pai_mubufload.inc"
};

const uint32_t pai_arith_code[PAI_ARITH_CODE_WORDS] = {
#include "gfx1013/pai_arith.inc"
};

const uint32_t pai_arith4_code[] = {
#include "gfx1013/pai_arith4.inc"
};

const uint32_t pai_fbatch_code[] = {
#include "gfx1013/pai_fbatch.inc"
};

const uint32_t pai_shotgun_code[PAI_SHOTGUN_CODE_WORDS] = {
#include "gfx1013/pai_shotgun.inc"
};

const uint32_t pai_vecscalar_code[PAI_VECSCALAR_CODE_WORDS] = {
#include "gfx1013/pai_vecscalar.inc"
};

const uint32_t pai_gbatch_code[] = {
#include "gfx1013/pai_gbatch.inc"
};

const uint32_t pai_gbatch2_code[] = {
#include "gfx1013/pai_gbatch2.inc"
};

const uint32_t pai_gbatch3_code[] = {
#include "gfx1013/pai_gbatch3.inc"
};

const uint32_t pai_g15_code[PAI_G15_CODE_WORDS] = {
#include "gfx1013/pai_g15.inc"
};
const uint32_t pai_selfref_code[PAI_G16_CODE_WORDS] = {
#include "gfx1013/pai_selfref.inc"
};
const uint32_t pai_acqload_code[PAI_G17_CODE_WORDS] = {
#include "gfx1013/pai_acqload.inc"
};
const uint32_t pai_selfref_nowait_code[PAI_G18_CODE_WORDS] = {
#include "gfx1013/pai_selfref_nowait.inc"
};

const uint32_t pai_hbatch_code[] = {
#include "gfx1013/pai_hbatch.inc"
};

const uint32_t pai_hbatch2_code[] = {
#include "gfx1013/pai_hbatch2.inc"
};

const uint32_t pai_hbatch3_code[PAI_H10_CODE_WORDS] = {
#include "gfx1013/pai_hbatch3.inc"
};

const uint32_t pai_hbatch4_code[PAI_H11_CODE_WORDS] = {
#include "gfx1013/pai_hbatch4.inc"
};

const uint32_t pai_hbatch5_code[] = {
#include "gfx1013/pai_hbatch5.inc"
};

const uint32_t pai_v0model_code[] = {
#include "gfx1013/pai_v0model.inc"
};

const uint32_t pai_loadstore_gold_code[PAI_LOADSTORE_GOLD_CODE_WORDS] = {
#include "gfx1013/pai_loadstore_gold.inc"
};

const uint32_t pai_loadstore_code[PAI_LOADSTORE_CODE_WORDS] = {
#include "gfx1013/pai_loadstore.inc"
};

const uint32_t pai_bisect_code[] = {
#include "gfx1013/pai_bisect.inc"
};
