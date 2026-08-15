#include "pm4.h"

void
pai_pm4_builder_init(pai_pm4_builder_t *b, uint32_t *buf, uint32_t cap) {
  b->buf = buf;
  b->cap = cap;
  b->len = 0;
}

uint32_t *
pai_pm4_emit(pai_pm4_builder_t *b, uint32_t dwords) {
  uint32_t *p;

  if (dwords == 0 || dwords > b->cap - b->len) {
    return NULL;
  }

  p = b->buf + b->len;
  b->len += dwords;
  return p;
}

uint32_t *
pai_pm4_nop(pai_pm4_builder_t *b, uint32_t dwords) {
  uint32_t *p;

  if (dwords < 2) {
    return NULL;
  }

  p = pai_pm4_emit(b, dwords);
  if (!p) {
    return NULL;
  }

  p[0] = pai_pm4_header3(PAI_PM4_OP_NOP, dwords);
  for (uint32_t i = 1; i < dwords; i++) {
    p[i] = 0;
  }
  return p;
}

uint32_t *
pai_pm4_set_sh_reg_compute(pai_pm4_builder_t *b, uint32_t reg_offset,
                           uint32_t count, const uint32_t *values) {
  uint32_t *p;

  if (count == 0 || count > 0x3FFE || !values) {
    return NULL;
  }

  p = pai_pm4_emit(b, count + 2);
  if (!p) {
    return NULL;
  }

  p[0] = pai_pm4_header3(PAI_PM4_OP_SET_SH_REG, count + 2) | 1u;
  p[1] = reg_offset & 0xFFFFu;
  for (uint32_t i = 0; i < count; i++) {
    p[2 + i] = values[i];
  }
  return p;
}

uint32_t *
pai_pm4_dispatch_direct(pai_pm4_builder_t *b, uint32_t group_x,
                        uint32_t group_y, uint32_t group_z,
                        uint32_t modifier) {
  uint32_t *p = pai_pm4_emit(b, 5);

  if (!p) {
    return NULL;
  }

  p[0] = pai_pm4_header3(PAI_PM4_OP_DISPATCH_DIRECT, 5) | 1u;
  p[1] = group_x;
  p[2] = group_y;
  p[3] = group_z;
  p[4] = (modifier & 0xA038u) | PAI_PM4_DISPATCH_INITIATOR;
  return p;
}

uint32_t *
pai_pm4_release_mem_eop(pai_pm4_builder_t *b, uint32_t event_type,
                        uint32_t event_index, uint64_t addr, uint32_t data) {
  uint32_t *p = pai_pm4_emit(b, 8);

  if (!p) {
    return NULL;
  }

  p[0] = pai_pm4_header3(PAI_PM4_OP_RELEASE_MEM, 8);
  p[1] = (event_type & 0x3Fu) | ((event_index & 0x0Fu) << 8);
  p[2] = (uint32_t)(addr & 0xFFFFFFFFu);
  p[3] = (uint32_t)(addr >> 32);
  p[4] = data;
  p[5] = 0;
  p[6] = 0;
  p[7] = 0;
  return p;
}

uint32_t *
pai_pm4_dma_data(pai_pm4_builder_t *b, uint64_t src, uint64_t dst,
                 uint32_t size) {
  uint32_t *p = pai_pm4_emit(b, 7);

  if (!p || size > 0x1FFFFFu) {
    return NULL;
  }

  p[0] = (3u << 30) | (1u << 1) | (PAI_PM4_OP_DMA_DATA << 8) | (5u << 16);
  p[1] = (2u << 13) | (1u << 15) | (2u << 25) | (1u << 27) | (1u << 31);
  p[2] = (uint32_t)(src & 0xFFFFFFFFu);
  p[3] = (uint32_t)(src >> 32);
  p[4] = (uint32_t)(dst & 0xFFFFFFFFu);
  p[5] = (uint32_t)(dst >> 32);
  p[6] = size & 0x1FFFFFu;
  return p;
}

uint32_t *
pai_pm4_indirect_buffer(pai_pm4_builder_t *b, uint64_t ib_addr,
                        uint32_t ib_size_dwords) {
  uint32_t *p = pai_pm4_emit(b, 4);

  if (!p || ib_size_dwords > 0xFFFFFu) {
    return NULL;
  }

  p[0] = pai_pm4_header3(PAI_PM4_OP_INDIRECT_BUFFER, 4);
  p[1] = (uint32_t)(ib_addr & 0xFFFFFFFFu);
  p[2] = (uint32_t)(ib_addr >> 32) & 0xFFFFu;
  p[3] = ib_size_dwords & 0xFFFFFu;
  return p;
}

uint32_t *
pai_pm4_wait_reg_mem(pai_pm4_builder_t *b, uint32_t compare_function,
                     uint32_t operation, uint32_t cache_policy,
                     uint64_t address, uint32_t reference, uint32_t mask,
                     uint32_t poll_cycles) {
  uint32_t *p = pai_pm4_emit(b, 7);

  if (!p || compare_function > 7 || operation > 4 || cache_policy > 3) {
    return NULL;
  }

  p[0] = pai_pm4_header3(PAI_PM4_OP_WAIT_REG_MEM, 7);
  p[1] = 0x10u | (compare_function & 0x7u) | ((operation & 0x3u) << 8u) |
         ((operation & 0xCu) << 4u) | ((cache_policy & 0x3u) << 25u);
  p[2] = (uint32_t)address & ~0x3u;
  p[3] = (uint32_t)(address >> 32) & 0x3FFFFu;
  p[4] = reference;
  p[5] = mask;
  p[6] = poll_cycles > 0xFFFF0 ? 0xFFFFu : poll_cycles >> 4u;
  return p;
}

uint32_t *
pai_pm4_event_write(pai_pm4_builder_t *b, uint32_t event_type,
                    uint32_t event_index) {
  uint32_t *p = pai_pm4_emit(b, 2);

  if (!p) {
    return NULL;
  }

  p[0] = pai_pm4_header3(PAI_PM4_OP_EVENT_WRITE, 2);
  p[1] = (event_type & 0x3Fu) | ((event_index & 0xFu) << 8);
  return p;
}
