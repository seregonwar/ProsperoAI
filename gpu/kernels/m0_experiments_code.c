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
const uint32_t pai_smemload_code[PAI_G19_CODE_WORDS] = {
#include "gfx1013/pai_smemload.inc"
};
const uint32_t pai_mubufload_clean_code[PAI_G20_CODE_WORDS] = {
#include "gfx1013/pai_mubufload_clean.inc"
};
const uint32_t pai_mubufload_g15_code[PAI_G21_CODE_WORDS] = {
#include "gfx1013/pai_mubufload_g15.inc"
};
const uint32_t pai_smemload_g15_code[PAI_G22_CODE_WORDS] = {
#include "gfx1013/pai_smemload_g15.inc"
};
const uint32_t pai_smemvecadd_code[PAI_G23_CODE_WORDS] = {
#include "gfx1013/pai_add1d.inc"
};
const uint32_t pai_saxpy_code[PAI_SAXPY_CODE_WORDS] = {
#include "gfx1013/pai_saxpy.inc"
};
const uint32_t pai_dot_serial_u32_code[PAI_DOT_SERIAL_U32_CODE_WORDS] = {
#include "gfx1013/pai_dot_serial_u32.inc"
};
const uint32_t pai_gemv_serial_u32_code[PAI_GEMV_SERIAL_U32_CODE_WORDS] = {
#include "gfx1013/pai_gemv_serial_u32.inc"
};
const uint32_t pai_smemload16_code[PAI_G24_CODE_WORDS] = {
#include "gfx1013/pai_smemload16.inc"
};
const uint32_t pai_dsprobe_code[PAI_G25_CODE_WORDS] = {
#include "gfx1013/pai_dsprobe.inc"
};
const uint32_t pai_lds_lanes_code[PAI_LDS_LANES_CODE_WORDS] = {
#include "gfx1013/pai_lds_lanes.inc"
};
const uint32_t pai_dsstaged_code[PAI_G26_CODE_WORDS] = {
#include "gfx1013/pai_dsstaged.inc"
};
const uint32_t pai_dsstaged1_code[PAI_G27_CODE_WORDS] = {
#include "gfx1013/pai_dsstaged1.inc"
};
const uint32_t pai_fbatch2_code[] = {
#include "gfx1013/pai_fbatch2.inc"
};
const uint32_t pai_fbatch3_code[] = {
#include "gfx1013/pai_fbatch3.inc"
};
const uint32_t pai_fbatch4_code[PAI_G33_CODE_WORDS] = {
#include "gfx1013/pai_fbatch4.inc"
};
const uint32_t pai_fbatch5_code[] = {
#include "gfx1013/pai_fbatch5.inc"
};
const uint32_t pai_mubufload36_code[PAI_G36_CODE_WORDS] = {
#include "gfx1013/pai_mubufload36.inc"
};
const uint32_t pai_mubufload37_code[] = {
#include "gfx1013/pai_mubufload37.inc"
};
const uint32_t pai_fdot_serial_code[PAI_FDOT_CODE_WORDS] = {
#include "gfx1013/pai_fdot_serial.inc"
};
const uint32_t pai_fgemv_serial_code[PAI_FGEMV_CODE_WORDS] = {
#include "gfx1013/pai_fgemv_serial.inc"
};
const uint32_t pai_fsaxpy_code[PAI_FSAXPY_CODE_WORDS] = {
#include "gfx1013/pai_fsaxpy.inc"
};

const uint32_t pai_t4_ops_code[PAI_T4_CODE_WORDS] = {
#include "gfx1013/pai_t4_ops.inc"
};

const uint32_t pai_int_ops_code[PAI_INT_CODE_WORDS] = {
#include "gfx1013/pai_int_ops.inc"
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
