#include "test.h"

#include <pm4/pm4.h>

#include <stdint.h>

/*
 * Golden byte checks against the hardware-qualified references:
 *  - OpenAGC cb_builders.c (SET_SH_REG / DISPATCH_DIRECT / RELEASE_MEM)
 *  - PS5-Firmware-Spoofer (raw IT_DMA_DATA, 11.20)
 */

TEST_MAIN_BEGIN()

{
  static uint32_t buf[64];
  pai_pm4_builder_t b;
  pai_pm4_builder_init(&b, buf, 64);

  /* SET_SH_REG, 2 regs, compute bank: header 0xC0027601 */
  uint32_t vals[2] = {0x11111111, 0x22222222};
  pai_pm4_set_sh_reg_compute(&b, 0x212, 2, vals);
  CHECK_EQ_UINT(buf[0], 0xC0027601u);
  CHECK_EQ_UINT(buf[1], 0x212);
  CHECK_EQ_UINT(buf[2], 0x11111111);
  CHECK_EQ_UINT(buf[3], 0x22222222);
  CHECK_EQ_UINT(b.len, 4);
}

{
  static uint32_t buf[64];
  pai_pm4_builder_t b;
  pai_pm4_builder_init(&b, buf, 64);

  /* DISPATCH_DIRECT: header 0xC0031501, initiator 0x41 */
  pai_pm4_dispatch_direct(&b, 2, 3, 4, 0);
  CHECK_EQ_UINT(buf[0], 0xC0031501u);
  CHECK_EQ_UINT(buf[1], 2);
  CHECK_EQ_UINT(buf[2], 3);
  CHECK_EQ_UINT(buf[3], 4);
  CHECK_EQ_UINT(buf[4], 0x41);
}

{
  static uint32_t buf[64];
  pai_pm4_builder_t b;
  pai_pm4_builder_init(&b, buf, 64);

  /* RELEASE_MEM EOP: header 0xC0064900, 8 dwords */
  pai_pm4_release_mem_eop(&b, 0x14, 0, 0x1122334455667788ULL, 0xDEADBEEF);
  CHECK_EQ_UINT(buf[0], 0xC0064900u);
  CHECK_EQ_UINT(buf[1], 0x14);
  CHECK_EQ_UINT(buf[2], 0x55667788u);
  CHECK_EQ_UINT(buf[3], 0x11223344u);
  CHECK_EQ_UINT(buf[4], 0xDEADBEEF);
  CHECK_EQ_UINT(buf[5], 0);
  CHECK_EQ_UINT(buf[6], 0);
  CHECK_EQ_UINT(buf[7], 0);
}

{
  static uint32_t buf[64];
  pai_pm4_builder_t b;
  pai_pm4_builder_init(&b, buf, 64);

  /* Raw IT_DMA_DATA (spoofer layout): header 0xC0055002, 7 dwords */
  pai_pm4_dma_data(&b, 0xAAAABBBBCCCCDDDDULL, 0x1111222233334444ULL,
                   0x80000);
  CHECK_EQ_UINT(buf[0], 0xC0055002u);
  CHECK_EQ_UINT(buf[1], 0x8C00C000u); /* (2<<13)|(1<<15)|(2<<25)|(1<<27)|(1<<31) */
  CHECK_EQ_UINT(buf[2], 0xCCCCDDDDu);
  CHECK_EQ_UINT(buf[3], 0xAAAABBBBu);
  CHECK_EQ_UINT(buf[4], 0x33334444u);
  CHECK_EQ_UINT(buf[5], 0x11112222u);
  CHECK_EQ_UINT(buf[6], 0x80000);
}

{
  static uint32_t buf[64];
  pai_pm4_builder_t b;
  pai_pm4_builder_init(&b, buf, 64);

  /* NOP trailer: 16 dwords, header 0xC00E1000 */
  pai_pm4_nop(&b, 16);
  CHECK_EQ_UINT(buf[0], 0xC00E1000u);
  for (int i = 1; i < 16; i++) {
    CHECK_EQ_UINT(buf[i], 0);
  }
}

{
  static uint32_t buf[64];
  pai_pm4_builder_t b;
  pai_pm4_builder_init(&b, buf, 64);

  /* Action-based EOP fence (OpenAGC runtime layout, FW 5.50-proven):
   * header 0xC0064900, gcr 0x703, data_sel 1, 8 dwords + 2-dword NOP. */
  pai_pm4_release_mem_eop_fence(&b, 0x1122334455667788ULL, 0xDEADBEEF);
  pai_pm4_nop(&b, 2);
  CHECK_EQ_UINT(buf[0], 0xC0064900u);
  CHECK_EQ_UINT(buf[1], 0x06703514u);
  CHECK_EQ_UINT(buf[2], 0x20000000u);
  CHECK_EQ_UINT(buf[3], 0x55667788u);
  CHECK_EQ_UINT(buf[4], 0x11223344u);
  CHECK_EQ_UINT(buf[5], 0xDEADBEEF);
  CHECK_EQ_UINT(buf[6], 0);
  CHECK_EQ_UINT(buf[7], 0);
  CHECK_EQ_UINT(buf[8], 0xC0001000u); /* trailing NOP */
  CHECK_EQ_UINT(buf[9], 0);
}

{
  static uint32_t buf[64];
  pai_pm4_builder_t b;
  pai_pm4_builder_init(&b, buf, 64);

  /* IT_WRITE_DATA: header 0xC0033700 for 1 data dword. */
  uint32_t data[1] = {0xCAFEBABE};
  pai_pm4_write_data(&b, 0x8877665544332210ULL, data, 1);
  CHECK_EQ_UINT(buf[0], 0xC0033700u);
  CHECK_EQ_UINT(buf[1], 0);
  CHECK_EQ_UINT(buf[2], 0x44332210u);
  CHECK_EQ_UINT(buf[3], 0x88776655u);
  CHECK_EQ_UINT(buf[4], 0xCAFEBABE);
}

{
  static uint32_t buf[64];
  pai_pm4_builder_t b;
  pai_pm4_builder_init(&b, buf, 64);

  /* WAIT_REG_MEM (FW 5.50 layout): header 0xC0053C00 */
  pai_pm4_wait_reg_mem(&b, 3, 2, 1, 0x8877665544332211ULL, 0xABCDEF01,
                       0xFFFFFFFF, 64);
  CHECK_EQ_UINT(buf[0], 0xC0053C00u);
  CHECK_EQ_UINT(buf[1], 0x10u | 3u | (2u << 8) | (0u << 4) | (1u << 25));
  CHECK_EQ_UINT(buf[2], 0x44332210u);
  CHECK_EQ_UINT(buf[3], 0x776655u & 0x3FFFFu);
  CHECK_EQ_UINT(buf[4], 0xABCDEF01);
  CHECK_EQ_UINT(buf[5], 0xFFFFFFFF);
  CHECK_EQ_UINT(buf[6], 4);
}

TEST_MAIN_END()
