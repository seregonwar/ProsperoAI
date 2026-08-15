# /dev/gc GPU submission ABI — FW 9.40 notes

Reverse-engineering notes for the PAI-M0 GPU bring-up path. Sources:
OpenAGC (Apache-2.0, FW 5.50-hardware-proven, mirrored at
`C:\Users\marco\AppData\Local\Temp\opencode\openagc-ref`), PS5-Firmware-Spoofer
(darkness/tcphdr, 11.20), kernel_940.elf string analysis.

## Confirmed on 9.40

- The `gc` kernel driver exists on 9.40 (strings in kernel_940.elf:
  `gc_open: reinitialize primary rings and micro codes`, `gc_submit_with_pid`,
  `gc_handle_user_release_mem_intr`, `gc_pm4_clearstate_patch`, `gc_switch`).
- OpenAGC's `driver_registry.c` states every inspected standard FW image
  (3.20-12.70) shares the normalized submit16 ioctl group, so 9.40 uses the
  same ABI as the 5.50 hardware proof and the 11.20 spoofer.

## Submission

```
open("/dev/gc", O_RDWR)

ioctl 0xC004812E            CONTEXT_QUERY, nr=0x2e, 4 bytes R  (optional probe)
ioctl 0xC0108102            SUBMIT_16,   nr=0x02, 16 bytes RW
```

Ioctl encoding (Sony custom): `(dir << 30) | (size << 16) | (0x81 << 8) | nr`
with dir = 1 W / 2 R / 3 RW. So `0xC0108102` = RW, 16 bytes, group 0x81, nr 2.

Submit arg (16 bytes):

```c
struct { uint32_t queue_type;  /* 3 = graphics queue (SPRX-confirmed) */
         uint32_t num_cbs;     /* 1..0xFFF, kernel-validated            */
         uint64_t cb_array;    /* user VA of descriptor array          */
};
```

Command-buffer descriptor (16 bytes each, copyin'd by the kernel):

```c
struct { uint64_t header;   /* [63:32]=ib_base_lo, [31:0]=0xC0023F00 */
         uint64_t ib_base;  /* [63:32]=ib_size_dwords, [31:0]=ib_base_hi */
};
```

The kernel masks `ib_base` with `0x000FFFFF0000FFFF`, validates the header
opcode (0x3F/0x33) and ORs the process VMID into bits [63:52] after copyin.

Payload-context completion: the GFX ring can defer the final descriptor, so
every submit appends a trailing 16-dword NOP indirect buffer (FW 5.50
hardware-proven payload completion sequence, OpenAGC `driver_prospero.c`).

## GPU memory

```c
sceKernelAllocateMainDirectMemory(len=2MB-granular, align=len, type=1, &phys);
sceKernelMapNamedDirectMemory(&va, len,
    PROT_READ|PROT_WRITE|PROT_GPU_READ(0x10)|PROT_GPU_WRITE(0x20),
    MAP_NO_COALESCE(0x400000), phys, align, "name");
```

GPU VA == CPU VA. All direct memory is GPU-visible; no separate VRAM.

## Packets (hardware-qualified layouts)

- Type-3 header: `0xC0000000 | ((count-2)&0x3FFF)<<16 | (op<<8)`;
  compute bank select = `header |= 1` for SET_SH_REG (0x76) and
  DISPATCH_DIRECT (0x15).
- IT_DISPATCH_DIRECT: x, y, z, `(mod & 0xA038) | 0x41`.
- IT_RELEASE_MEM EOP fence (OpenAGC sceAgcDcbSetEopFlip, 8 dwords):
  `0xC0064900`, `event_type[5:0] | event_index[13:8]`, addr_lo, addr_hi,
  data, 0, 0, 0. Event type 0x14 = cache-flush event.
- Raw IT_DMA_DATA (spoofer, 7 dwords): header `(3<<30)|(1<<1)|(0x50<<8)|(5<<16)`,
  ctrl `(2<<13)|(1<<15)|(2<<25)|(1<<27)|(1<<31)`, src_lo/hi, dst_lo/hi,
  size&0x1FFFFF.
- IT_WAIT_REG_MEM (FW 5.50, 7 dwords): `0xC0053C00`, ctrl
  `0x10 | cmp&7 | (op&3)<<8 | (op&0xC)<<4 | (cache&3)<<25`, addr_lo&~3,
  addr_hi&0x3FFFF, ref, mask, poll=min(poll_cycles>>4, 0xFFFF).

## Shader dispatch (gfx1013)

- Registers: PGM_LO/HI = code addr >> 8 / >> 40; RSRC1 = 0x602C0000
  (WGP_MODE|W32_EN|float-mode, from OpenAGC's hardware-qualified memset
  kernel); RSRC2 USER_SGPR[5:1] = user SGPR count; RSRC3 = 0;
  NUM_THREAD_X/Y/Z; USER_DATA_0..15.
- No preamble needed: firmware register defaults cover the compute state
  (SE0-3 destination enable 0xFFFFFFFF, DISPATCH_TUNNEL 0x4FF, CHKSUM 0).
- gfx1013 = ISA 10.1.3: no VOPD, no dot-product instructions; VOP3 encoding
  is the standard gfx10 form. Hand-written kernels assemble with
  `llvm-mc -triple=amdgcn -mcpu=gfx1013` (LLVM >= 14).
- v_addc_co_u32 is not available on gfx1013; use v_add_co_ci_u32.

## Open questions

- Exact gc ioctl handler layout on 9.40 (SUB MIT_16 assumed identical per
  OpenAGC's firmware-range statement; to be validated at runtime).
- Whether the /dev/gc device file is present under all 9.40 payload hosts
  (probe at runtime; capability flag decides).
- EOP label polling semantics on 9.40 (OpenAGC polls dmem labels for fences).

## References

- OpenAGC: https://github.com/OpenAGC/OpenAGC (Apache-2.0)
  - include/agc_ioctl.h, src/cb_builders.c, src/dcb.c, src/driver_prospero.c,
    src/driver_registry.c, src/memset_exclusive_shader.h
- PS5-Firmware-Spoofer: https://github.com/tcphdr/PS5-Firmware-Spoofer
- KytyPS5: https://github.com/KytyPS5/KytyPS5 (register semantics)
- shadPS4: https://github.com/shadps4-emu/shadPS4 (GNM semantics, PS4)
