/*
 * ProsperoAI — PM4 command stream builder
 *
 * PM4 type-3 packet construction for the PS5 GPU command processor.
 * Encodings follow the hardware-qualified references:
 *   - OpenAGC cb_builders.c / agc_ioctl.h (FW 5.50-proven, shared by
 *     every standard FW 3.20-12.70 including 9.40)
 *   - PS5-Firmware-Spoofer (raw IT_DMA_DATA on 11.20)
 *
 * Every packet is emitted through a cursor struct so the same streams
 * are consumed by the real /dev/gc backend and the host reference
 * interpreter (gpu/hal/backend_host_ref.c).
 */

#ifndef PAI_PM4_H
#define PAI_PM4_H

#include <stdint.h>

/* PM4 type-3 opcodes used by PAI-M0. */
#define PAI_PM4_OP_NOP              0x10
#define PAI_PM4_OP_DISPATCH_DIRECT  0x15
#define PAI_PM4_OP_CONTEXT_CONTROL  0x28
#define PAI_PM4_OP_INDIRECT_BUFFER  0x3F
#define PAI_PM4_OP_ACQUIRE_MEM      0x58
#define PAI_PM4_OP_EVENT_WRITE      0x46
#define PAI_PM4_OP_RELEASE_MEM      0x49
#define PAI_PM4_OP_DMA_DATA         0x50
#define PAI_PM4_OP_SET_CONFIG_REG   0x68
#define PAI_PM4_OP_SET_SH_REG       0x76
#define PAI_PM4_OP_WAIT_REG_MEM     0x3C
#define PAI_PM4_OP_WRITE_DATA       0x37

/* Compute shader registers (SH bank, compute pipeline). */
#define PAI_REG_COMPUTE_START_X        0x0204
#define PAI_REG_COMPUTE_START_Y        0x0205
#define PAI_REG_COMPUTE_START_Z        0x0206
#define PAI_REG_COMPUTE_NUM_THREAD_X   0x0207
#define PAI_REG_COMPUTE_NUM_THREAD_Y   0x0208
#define PAI_REG_COMPUTE_NUM_THREAD_Z   0x0209
#define PAI_REG_COMPUTE_PGM_LO         0x020C
#define PAI_REG_COMPUTE_PGM_HI         0x020D
#define PAI_REG_COMPUTE_PGM_RSRC1      0x0212
#define PAI_REG_COMPUTE_PGM_RSRC2      0x0213
#define PAI_REG_COMPUTE_PGM_RSRC3      0x0228
#define PAI_REG_COMPUTE_USER_DATA_0    0x0240

/*
 * COMPUTE_PGM_RSRC2 field layout (gfx10 / PS5 Gen5).
 *
 * Bit positions and masks rewritten in C from Kyty's MIT-licensed
 * Pm4.h (Copyright (c) 2021 Ivan Chikhradze). LDS_SIZE is in 512-byte
 * granules: lds_bytes = LDS_SIZE * 512. AMD ISA: bit 6 is TRAP_PRESENT
 * (Kyty's listing skips it; historical PAI G25 0x4C set that bit and
 * left LDS_SIZE=0).
 */
#define PAI_COMPUTE_PGM_RSRC2_SCRATCH_EN_SHIFT     0
#define PAI_COMPUTE_PGM_RSRC2_SCRATCH_EN_MASK      0x1u
#define PAI_COMPUTE_PGM_RSRC2_USER_SGPR_SHIFT      1
#define PAI_COMPUTE_PGM_RSRC2_USER_SGPR_MASK       0x1Fu
#define PAI_COMPUTE_PGM_RSRC2_TRAP_PRESENT_SHIFT   6
#define PAI_COMPUTE_PGM_RSRC2_TRAP_PRESENT_MASK    0x1u
#define PAI_COMPUTE_PGM_RSRC2_TGID_X_EN_SHIFT      7
#define PAI_COMPUTE_PGM_RSRC2_TGID_X_EN_MASK       0x1u
#define PAI_COMPUTE_PGM_RSRC2_TGID_Y_EN_SHIFT      8
#define PAI_COMPUTE_PGM_RSRC2_TGID_Y_EN_MASK       0x1u
#define PAI_COMPUTE_PGM_RSRC2_TGID_Z_EN_SHIFT      9
#define PAI_COMPUTE_PGM_RSRC2_TGID_Z_EN_MASK       0x1u
#define PAI_COMPUTE_PGM_RSRC2_TG_SIZE_EN_SHIFT     10
#define PAI_COMPUTE_PGM_RSRC2_TG_SIZE_EN_MASK      0x1u
#define PAI_COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_SHIFT 11
#define PAI_COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_MASK  0x3u
#define PAI_COMPUTE_PGM_RSRC2_LDS_SIZE_SHIFT       15
#define PAI_COMPUTE_PGM_RSRC2_LDS_SIZE_MASK        0x1FFu
#define PAI_COMPUTE_PGM_RSRC2_LDS_GRANULE_BYTES    512u

#define PAI_RSRC2_USER_SGPR(n)                                                     \
  ((((uint32_t)(n)) & PAI_COMPUTE_PGM_RSRC2_USER_SGPR_MASK)                        \
   << PAI_COMPUTE_PGM_RSRC2_USER_SGPR_SHIFT)
#define PAI_RSRC2_TGID_X_EN                                                        \
  (PAI_COMPUTE_PGM_RSRC2_TGID_X_EN_MASK << PAI_COMPUTE_PGM_RSRC2_TGID_X_EN_SHIFT)
#define PAI_RSRC2_LDS_GRANULES(n)                                                  \
  ((((uint32_t)(n)) & PAI_COMPUTE_PGM_RSRC2_LDS_SIZE_MASK)                         \
   << PAI_COMPUTE_PGM_RSRC2_LDS_SIZE_SHIFT)

/* Default dispatch initiator (OpenAGC: (mod & 0xA038) | 0x41). */
#define PAI_PM4_DISPATCH_INITIATOR     0x41u

/* RELEASE_MEM EOP event: cache-flush event type used for fences. */
#define PAI_GFX1013_EOP_CACHE_FLUSH_EVENT 0x14u

/* OpenAGC native-runtime fence constants (FW 5.50-hardware-proven). */
#define PAI_GFX1013_EOP_GCR_CONTROL   0x703u
#define PAI_GFX1013_EOP_CACHE_POLICY  3u
#define PAI_GFX1013_EOP_EVENT_INDEX   5u
#define PAI_GFX1013_EOP_DATA_SEL_32B  1u

typedef struct pai_pm4_builder {
  uint32_t *buf;
  uint32_t  cap;
  uint32_t  len;
} pai_pm4_builder_t;

void pai_pm4_builder_init(pai_pm4_builder_t *b, uint32_t *buf, uint32_t cap);

/* Reserve `dwords` and return the start pointer, or NULL on overflow. */
uint32_t *pai_pm4_emit(pai_pm4_builder_t *b, uint32_t dwords);

static inline uint32_t
pai_pm4_header3(uint32_t op, uint32_t count) {
  return 0xC0000000u | (((count - 2) & 0x3FFFu) << 16) | (op << 8);
}

/* NOP packet (also used as the completion trailer). */
uint32_t *pai_pm4_nop(pai_pm4_builder_t *b, uint32_t dwords);

/*
 * SET_SH_REG with the compute-bank select bit set (bit 0 of the header),
 * matching OpenAGC sceAgcCbMemsetExclusive.
 */
uint32_t *pai_pm4_set_sh_reg_compute(pai_pm4_builder_t *b, uint32_t reg_offset,
                                     uint32_t count, const uint32_t *values);

/* IT_DISPATCH_DIRECT with compute-bank bit set. */
uint32_t *pai_pm4_dispatch_direct(pai_pm4_builder_t *b, uint32_t group_x,
                                  uint32_t group_y, uint32_t group_z,
                                  uint32_t modifier);

/*
 * IT_CONTEXT_CONTROL (0x28): enables context-register shadowing so the
 * compute register state commits (OpenAGC agcGfx1013SetContextControl,
 * 0x80000000/0x80000000 = load/shadow enable, hardware-proven).
 */
uint32_t *pai_pm4_context_control(pai_pm4_builder_t *b, uint32_t load_control,
                                  uint32_t shadow_control);

/*
 * IT_ACQUIRE_MEM (0x58): invalidate the GPU caches so the shader reads
 * data the CPU wrote (OpenAGC agcGfx1013EmitAcquireAll, GCR_ALL=0xC3B1).
 * Must precede the first shader load of host-written buffers.
 */
uint32_t *pai_pm4_acquire_mem(pai_pm4_builder_t *b);

/*
 * IT_RELEASE_MEM EOP fence (OpenAGC sceAgcDcbSetEopFlip layout, 8 dwords).
 * Legacy bring-up variant; prefer pai_pm4_release_mem_eop_fence().
 */
uint32_t *pai_pm4_release_mem_eop(pai_pm4_builder_t *b, uint32_t event_type,
                                  uint32_t event_index, uint64_t addr,
                                  uint32_t data);

/*
 * IT_RELEASE_MEM EOP fence — the action-based layout the real driver
 * uses for GPU->host fences (OpenAGC sceAgcCbReleaseMem + the runtime's
 * agcGfx1013TransitionResource, hardware-qualified on FW 5.50):
 *
 *   [0] header 0xC0064900
 *   [1] event_type[5:0] | event_index[11:8] | gcr_control[23:12]
 *       | cache_policy[26:25]
 *   [2] destination[17:16] | interrupt[26:24] | data_selection[31:29]
 *   [3..4] address lo/hi (4-byte aligned)
 *   [5..6] data lo/hi (data_selection 1 writes the 32-bit value)
 *   [7] interrupt context id
 *
 * Must be followed by a 2-dword NOP (pai_pm4_nop), exactly like the
 * reference implementation.
 */
uint32_t *pai_pm4_release_mem_eop_fence(pai_pm4_builder_t *b, uint64_t addr,
                                        uint32_t value);

/*
 * IT_WRITE_DATA (0x37) — plain GPU memory write (OpenAGC
 * sceAgcDcbWriteData, SPRX-confirmed). Fallback completion signal that
 * needs no event machinery.
 */
uint32_t *pai_pm4_write_data(pai_pm4_builder_t *b, uint64_t addr,
                             const uint32_t *data, uint32_t dwords);

/*
 * Raw IT_DMA_DATA copy (7 dwords, hardware-proven layout from the
 * PS5-Firmware-Spoofer). Copies `size` bytes src -> dst, size <= 2MB-1.
 */
uint32_t *pai_pm4_dma_data(pai_pm4_builder_t *b, uint64_t src, uint64_t dst,
                           uint32_t size);

/* IT_INDIRECT_BUFFER. */
uint32_t *pai_pm4_indirect_buffer(pai_pm4_builder_t *b, uint64_t ib_addr,
                                  uint32_t ib_size_dwords);

/*
 * Native IT_WAIT_REG_MEM (32-bit compare, 7 dwords, FW 5.50 layout).
 * compare_function 0-7, operation 0-4, cache_policy 0-3.
 */
uint32_t *pai_pm4_wait_reg_mem(pai_pm4_builder_t *b, uint32_t compare_function,
                               uint32_t operation, uint32_t cache_policy,
                               uint64_t address, uint32_t reference,
                               uint32_t mask, uint32_t poll_cycles);

/* IT_EVENT_WRITE (2 dwords). */
uint32_t *pai_pm4_event_write(pai_pm4_builder_t *b, uint32_t event_type,
                              uint32_t event_index);

#endif /* PAI_PM4_H */
