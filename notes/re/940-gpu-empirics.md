# PS5 GPU (9.40) — Empirical silicon rules from PAI-M0 bring-up

Every rule below was verified on physical hardware (FW 9.40) through the
autonomous deploy loop (toolchain/deploy-test.ps1). Do NOT trust
OpenAGC's documentation blindly: several claims (bit15 semantics, user
data ABI, "hardware-qualified" memset) did not hold on 9.40.

## Submission (proven working)

- `/dev/gc` open + CONTEXT_QUERY ioctl (0xC004812E): caps = 0x00000000
  on 9.40. The SPRX-confirmed recovery step (mmap GPU register space at
  0xFE0200000, PROT 0x22, MAP_SHARED on the gc fd) runs but caps stay 0.
- SUBMIT_16 ioctl (0xC0108102): queue_type = 3 works (0 also tried;
  3 is the one that executed).
- CB descriptor {0xC0023F00 header | ib_lo<<32, ib_size<<32 | ib_hi}
  + 16-dword NOP trailer = correct on 9.40.
- IT_DMA_DATA (spoofer layout) executes and verifies.
- **Fence**: action-based IT_RELEASE_MEM EOP (OpenAGC runtime layout,
  event 0x14, index 5, gcr 0x703, cache 3, data_sel 1, + 2-dword NOP)
  fires. The SetEopFlip legacy layout does NOT.
- gc reopen between experiments (pai_gpu_reset) works — a wedged ring
  does not poison the next experiment.

## Compute dispatch (empirical)

- SET_SH_REG (compute bank bit |1) + DISPATCH_DIRECT (|1, initiator
  0x41) launches waves. The full OpenAGC preamble (CONTEXT_CONTROL
  0x80000000, RESOURCE_LIMITS, dest enables, START_X/Y/Z) is NOT
  required on 9.40.
- RSRC1 = 0x602C0000 (WGP_MODE + W32_EN) executes; RSRC2 counts from
  4 to 9 user SGPRs all dispatch.
- **v0 = per-thread id** (0..31 in a wave; verified with 32-thread
  groups writing 32 distinct addresses/values). The "v9 = tid" theory
  from the OpenAGC kernel disassembly was WRONG (v9 = 0 uniformly).
- Wave size = 32 (W32 mode) with WGP_MODE set.

## FLAT memory ops (the hard-won rules)

1. **vaddr pair must be 1 (v[2:3])**. flat ops with vaddr pair 2
   (v[4:5]) HANG the wave (E1-E7, E12/E13, E19-E21 all hung with
   pair 2; G3/G5 mutation bisect isolated word1 bits [2:1]: pair 2 =
   hang, pair 1 = runs). Data register field is irrelevant.
2. **flat_store_dwordx4 broadcasts v0** ({v0,v0,v0,v0}), ignoring the
   encoded data registers (E32: constant in v0 -> 4 copies stored).
   This is why OpenAGC's "memset" writes its offset ramp: its v0 held
   the computed offset. On 9.40 that kernel CANNOT write a real
   pattern — OpenAGC's memset claim is broken on this firmware.
3. **flat_store_dword (op 0x70) with pair 1 works** as a 4-byte store
   of v0 (E36: 32 correct per-thread 4-byte writes). (Early failures
   of op 0x70 were the pair-2 issue, not the opcode.)
4. **bit15 of FLAT word0 is irrelevant** (G2: golden kernel with bit15
   cleared still executes and writes).
5. **flat_load_dword HANGS on 9.40** — every variant tried (dst v0/v1/
   v6, pair 1) hung. The golden kernel uses SMEM loads, not flat loads.
   MUBUF loads (buffer_load_dword) dispatch but the T# format is still
   unresolved (loaded zeros with the first candidate).

## Value-path arithmetic (OPEN PROBLEM)

- E36/E30: v0 (tid) and constants store correctly; per-thread
  addressing is correct in every executing kernel.
- E39/E41/E42/E45: `v_add_f32 v0, v1, s4` (or with k moved to a VGPR)
  computes as if v1 = 0 for ALL threads — the address path sees the
  per-thread v1, the value path does not. Same register, two uses.
  Theories tried and falsified: mixed VGPR+SGPR VOP3 operands, odd
  user-SGPR counts, v9-as-tid, stride overlaps, 16-byte dword stores.
- E46 (v_add_f32 v0, v0, s0) HANGS — src0 = v0 in arithmetic VOPs is
  toxic on this silicon.
- Next: bisect the add with llvm-mc-assembled variants (e32 vs e64
  encodings), then try v_add_co_u32-style arithmetic, then MUBUF loads
  for the vecadd milestone. Hand-encoded kernels (E40, E48-E50) hang —
  do NOT hand-encode; always assemble with llvm-mc 18.

## Toolchain

- llvm-mc 18 (Windows, ps5-payload-sdk/tools/llvm18/bin) assembles
  gfx1013; llvm-mc 14 (WSL) produces identical encodings for the
  tested subset. ACO's kernel contains instructions LLVM's gfx1013
  model does not know (v_lshl_add_u32 etc.) — LLVM's model is
  conservative, the silicon is fuller RDNA2.
- llvm-objdump 14/18 cannot decode the OpenAGC kernel bytes (they use
  encodings outside LLVM's tables).

## Kernel RE of kernel_940.elf

- gc driver code ~VA 0x72F000-0x739000; ioctl dispatcher found
  (dir switch at +0x1977, jump table +0x1a8c); the kernel parses CBs
  and has special paths for PM4 opcodes 0xB5/0xEF/0xCA. gc_suspend/
  resume events appear in klog for system processes.
- Klog available via FTP /data/klog/klog.log.

## Status

- Working: bootstrap, jailbreak, /data logging, lifecycle listener,
  notify, DMA, fence, dispatch, stores, per-thread addressing.
- Blocked: arithmetic value path (see above) and loads.
- PAI-M0 milestone (GPU compute verified vs CPU) is one bisect away.
