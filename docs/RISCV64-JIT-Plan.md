# RISC-V 64 JIT backend: audit and plan

Status: phase 1 in progress (encoder + backend scaffold).

## 1. Audit of the existing JIT architecture

Boxedwine has one host-independent JIT front end and three host backends:

| Backend | Files | Size | Assembler |
|---|---|---|---|
| x86 / x64 host | `source/emulation/cpu/x32/jitX86CodeGen.cpp` | ~6.4k lines | asmjit x86 |
| ARMv8 host | `source/emulation/cpu/armv8/jitArmV8CodeGen.cpp` | ~7.5k lines | asmjit a64 |
| WebAssembly | `source/emulation/cpu/wasm/*` | ~18k lines | custom emitter |
| ARMv7 host | `source/emulation/cpu/armv7/armv7CPU.cpp` | dead code | targets the removed `dynamic/` recompiler, never compiled (`BOXEDWINE_DYNAMIC_ARMV7`) |

### Front end (`source/emulation/cpu/jit/`)

* `Jit` (`jit.h`): about 150 pure-virtual codegen primitives (register
  access, ALU ops, shifts and rotates, mul/div, structured `If*/StartElse/EndIf`,
  host calls, CPU struct access, lazy-flag storage, "direct" flag fusion).
  Every x86 instruction handler (`jitArith.h`, `jitMov.h`, `jitStrings.h`, ...)
  is written once against these primitives.
* `JitCodeGen` (`jitCodeGen.h/.cpp`): block discovery
  (`calculateLongestBlock`), the compile loop, block commit, block chaining
  (`blockNext1/2`, `jumpToEipIfCached`), the guest memory access paths
  (`read`/`write`/`readWriteMem`), lazy-flag materialization and the
  trampolines (`createEmulateSingleOp`, `createCalculationCF`, ...).
  A backend adds about 30 more hooks: buffer/label management
  (`markBufferLocation`, `getBufferLocation`, `copyBuffer`), `readMMU`,
  `readHost`/`writeHost`, `createStartJITCode`, `createBlockExit` and
  `createSyncToHost/FromHost`.
* `JitFPU` → `JitMMX` → `JitSSE` add x87/MMX/SSE primitives. Any SSE/MMX/x87
  op a backend does not override falls back to `emulateSingleOp()` (see
  `jitCallSSE.h`, `jitCallMMX.h`, `jitCallFpu.h`). **So a new backend can
  derive straight from `JitCodeGen` and still run everything correctly.**

### Memory-access configurations

`JitCodeGen::read/write` support three configurations:

1. **Portable (no `BOXEDWINE_HOST_EXCEPTIONS`, no `BOXEDWINE_MEM_CACHE`)**:
   every guest access does `mmu[addr >> 12]`, checks a permission bit and
   falls back to `emulateSingleOp` on failure. No signal handlers are needed.
   The 32-bit x86 build uses this.
2. **`BOXEDWINE_MEM_CACHE`**: a read/write cache table indexed like the MMU.
3. **`BOXEDWINE_HOST_EXCEPTIONS`**: linear memory with direct host loads and
   stores. Faults are recovered through a SIGSEGV handler
   (`platform/linux/platformThreads-{x64,armv8}.cpp`) using
   `jitAddressToEip`. This is the fastest mode, but it needs a per-arch
   `ucontext` handler.

The RISC-V backend starts in configuration 1 and moves to 3 in a later phase.
Today only the 32-bit x86 build exercises configuration 1. The x64 backend
does not compile without `BOXEDWINE_HOST_EXCEPTIONS` (`jitX86CodeGen.cpp` uses
`DecodedOp::exceptionCount`), so the RISC-V backend becomes the first 64-bit
user of the portable path.

### Integration points a new backend must touch

* `include/boxedwine.h`, `buildFlags.txt`: new `BOXEDWINE_JIT_RISCV64` flag.
* `source/emulation/cpu/common/cpu.h`: per-backend scratch in `CPU`
  (`storedRegs` for callee-saved host registers).
* `project/linux/makefile`: `uname -m == riscv64` branch plus a cross
  `test-riscv64` target.
* `source/test/testMain.cpp`, `source/test/cpu/testCPU.cpp`: backend-specific
  test guards.
* `clearJitBlock()`: each backend provides it (a no-op without host exceptions,
  same as x86).
* Later: `platform/linux/platformThreads-riscv64.cpp` (signal handler),
  `softmmu/soft_ram.cpp` linear-memory guards, `codePageData` entry cache.

## 2. RISC-V specific design decisions

* **ISA baseline: RV64GC + Zba + Zbb** (RVA22U64 profile). Zba (`add.uw`,
  `sh1add`...) gives cheap zero extension and scaled-index addressing. Zbb
  (`rev8`, `clz`, `ctz`, `rol/ror`, `sext.b/h`, `zext.h`, `andn`) covers BSWAP,
  BSF/BSR and the rotates. Every RVA22 board (VisionFive 2 is RVA20+Zba/Zbb,
  SpacemiT K1/X60, SG2042) has these. A plain-RV64GC fallback for each Zb*
  instruction can be added later behind a runtime check.
* **No assembler library.** asmjit has no RISC-V target. A small header-only
  encoder (`source/emulation/cpu/riscv64/rv64Emitter.h`) emits the ~80
  instructions needed, with labels and fixups. It is unit-tested against GNU
  `as` output.
* **Register-value invariant:** a host register holding a 32-bit guest value
  is always **zero-extended** to 64 bits. RISC-V `*W` instructions sign-extend,
  so 32-bit ALU results get `zext.w` (`add.uw rd, rs, zero`). This keeps
  `DYN_PTR`-width address math (`base + guestAddr`) correct. 8/16-bit partial
  register writes (AL/AH/AX) use insert sequences (`andi`/mask + `or`). There
  are no bitfield-insert instructions in the base ISA or Zb*.
* **No flags register.** RISC-V has no NZCV, so the "direct" flag fusion path
  (`direct_cmp` + `direct_jump` etc.) is disabled initially
  (`supportsDirectCondition()` returns false and lazy flags are always used).
  Phase 3 re-enables it by keeping the compare operands in registers and
  emitting fused compare-and-branch (`beq/bne/blt/bltu/bge/bgeu`), which maps
  very well onto x86 `cmp; jcc`.
* **Branch range.** Conditional branches reach ±4 KiB and `jal` ±1 MiB. Phase 1
  emits forward `If` branches as `b<inverted> +8; jal x0, label`, which is
  always correct. Later, a relaxation pass can shrink in-range branches.
* **Register map (LP64 ABI):** guest EAX..EDI cached in callee-saved
  `s2..s9` (x18..x25). CPU* in `s1` (x9), MMU base in `s10` (x26),
  `s11` (x27) holds the `emulateSingleOp` entry. The lazy-flag registers
  (flags, src, dst, result, type) and temps live in `a*`/`t*` and are written
  back around host calls by the `syncToHost/FromHost` stubs, as ARMv8 does.
* **Memory ordering.** RVWMO is weaker than x86 TSO. The ARMv8 backend
  defaults to no fences unless the hardware has TSO. RISC-V mirrors that:
  use Ztso when the kernel reports it, otherwise rely on `LOCK`-prefixed ops
  being emitted as AMOs/LR-SC with `.aqrl`.

## 3. Phased plan

**Phase 0: infrastructure (done in this change)**
- [x] Audit (this document)
- [x] riscv64 cross toolchain + qemu-user test loop: Ubuntu
      `g++-riscv64-linux-gnu`, `qemu-user`, riscv64 multiarch SDL2/GL/zlib
- [x] makefile: any `Build/Test*` dir builds the unit-test sources

**Phase 1: encoder + correct integer backend (this change, in progress)**
- [x] `rv64Emitter.h`: RV64I/M/A + Zba/Zbb encoder, labels, branch fixups,
      64-bit constant materialization; tests against GNU `as`
- [ ] `BOXEDWINE_JIT_RISCV64` flag wiring and makefile cross target
- [ ] `JitRiscv64CodeGen : JitCodeGen`: register allocation, all integer
      primitives, `If*` control flow, host calls, CPU access, MMU read path,
      start/exit/sync trampolines
- [ ] All existing CPU unit tests pass under `qemu-riscv64` (SSE/MMX/x87 run
      through `emulateSingleOp`)

**Phase 2: performance basics**
- `BOXEDWINE_HOST_EXCEPTIONS` + linear memory: SIGSEGV/SIGILL handler in
  `platform/linux/platformThreads-riscv64.cpp`, `soft_ram.cpp` guards
- Block-entry invalidation (`clearJitBlock` writes an illegal instruction)
- Native `LOCK` ops via AMO/LR-SC (`cmpxchg`, `xadd`, `xchg`, `lock add`...)
- Branch relaxation

**Phase 3: flags**
- Direct `cmp/test + jcc/setcc/cmovcc` fusion with compare-and-branch
- SF/ZF/CF fast paths without calling `calculateCF` helpers

**Phase 4: floating point and SIMD**
- x87 on F/D registers through the JitFPU interface (note: x87 80-bit is
  already modelled as 64-bit double by the other backends)
- SSE/SSE2 scalar via F/D. Packed SSE via RVV 1.0 when present, otherwise
  keep `emulateSingleOp`.

**Phase 5: shipping**
- Native riscv64 CI job (or a qemu job in Jenkins), `.deb` packaging, docs

## 4. Testing

```
# one-time (Ubuntu 24.04 host)
apt-get install qemu-user g++-riscv64-linux-gnu
dpkg --add-architecture riscv64   # with ports.ubuntu.com riscv64 sources
apt-get install libsdl2-dev:riscv64 libssl-dev:riscv64 zlib1g-dev:riscv64 libgl-dev:riscv64 libcurl4t64:riscv64

cd project/linux
make test-riscv64                  # cross build Build/TestRiscv64/boxedwine
qemu-riscv64 -L /usr/riscv64-linux-gnu Build/TestRiscv64/boxedwine
```
