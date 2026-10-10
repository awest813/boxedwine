# WASM JIT: audit and plan

Scope: `source/emulation/cpu/wasm/` (the `BOXEDWINE_WASM_JIT` backend, about 18k
lines), audited against the shared front end in `source/emulation/cpu/jit/`.
Items marked **verified** were reproduced with the Emscripten unit tests (see
Testing). The other items come from reading the code and still need measurement.

## How it works

* `JitWasmCodeGen : JitSSE`. The shared front end drives it, the same as the
  native backends. Each guest block becomes a wasm function
  `execute(cpu_ptr, relocBase)`. Guest registers, XMM registers and an x87 f64
  cache live in wasm locals and are synced back at exits and helper calls.
* Compilation is synchronous (`new WebAssembly.Module/Instance` in EM_JS). The
  function is installed with `addFunction`, and the table index goes into
  `DecodedOp::pfnJitCode`. `wasmStartJITOp` calls it through `call_indirect`.
* Blocks become hot after `JIT_RUN_COUNT` (200) runs. In single-threaded builds,
  file-backed blocks are batched (up to 64 blocks / 512 KB) and merged into one
  module (`wasmModuleMerger.cpp`). Tiny blocks wait for 600 more hits.
* In multi-threaded builds, table slots come from a shared atomic counter, each
  worker lazily installs modules, and a JS broker
  (`project/emscripten/boxedwine-wasm-jit-module-broker.js`) shares compiled
  modules across workers.
* Native code covers integer ALU, string ops (structured loops), an inline TLB
  memory fast path, x87 (an f64 cache with exact extended conversion), and
  MMX/SSE/SSE2 via SIMD128. Lazy flags stay in CPU memory. Z/NZ and logic
  conditions are inlined; other conditions call a helper.

## Key difference from the native backends

On x86/x64/ARMv8, `emulateSingleOp()` is a **tail jump** into the interpreter
trampoline, so code after it in the same op never runs. On WASM it is a
**call that returns**. The block continues after the helper unless it hits a
bailout (self-modifying code or control-flow redirect), a debug trap, or an
active x87 cache. Any front-end pattern of the form

```c++
If(slowPathNeeded) { emulateSingleOp(); } EndIf();
<native implementation of the same op>
```

runs the op twice on WASM. `guardCachedFpuFallback` already ends this pattern
with `blockExit()`.

## Findings

1. **Fixed (verified).** SSE `divps`/`divss` executed twice when MXCSR divide
   exceptions were unmasked (`JitSSE::guardSseDiv`). 8/2 gave 2 instead of 4.
   The guard now ends the block after the interpreter runs the op, the same as
   `guardCachedFpuFallback`. Regression tests: "Test SSE divide fast path
   decision" now executes non-faulting `divps`/`divss` with ZE unmasked.
   "Test x87 divide fast path decision" does the same for `fdivp`, which
   already worked on WASM because its x87 cache path ends the block.
2. **Front-end CS base assumption (fixed, found through the RISC-V work).**
   `Jit::writeCurrentEip` baked in the compile-time CS base. Signal delivery
   loads the flat code selector, so a handler inside the same block wrote a
   stale EIP. It now uses the live base when the process has set a CS base.
3. **MT LOCK ops always go through the interpreter.** The decoder rewrites
   `lock`-prefixed ops to `*_lock` instructions (`decoder.cpp`), and
   `jit/jitLock.h` emulates every one of them. The WASM emitter has no atomic
   (0xFE prefix) opcodes. Each `lock xadd/cmpxchg/inc` costs a full register
   sync, a helper call, lost lazy flags and a reload. These are hot in Wine
   (Interlocked*, critical sections).
4. **Interpreter fallback families** (`jitWasmCodeGen.h`, "fallbackToEmulateSingleOp"):
   rcl/rcr, 8/16-bit div/idiv, 8/16-bit imul forms, xadd/cmpxchg on 8/16-bit
   memory, cmpxchg8b, pusha/popa. Each one also resets the compile-time lazy
   flag type to `FLAGS_NULL`, which forces the condition helper on the next Jcc.
   `BOXEDWINE_WASM_JIT_FALLBACK_STATS` measures them.
5. **Branch hints are dropped in merged modules.** `wasmModuleMerger.cpp`
   passes through only the `name` custom section, so the
   `metadata.code.branch_hint` section written by `wasmEmitter.cpp` is lost
   for batched (file-backed) code, which is most code.
6. **Robustness.** Exhausting the scratch-local pool kpanics instead of falling
   back (`allocScratch` and friends). `callHostFunction` would emit a
   `call_indirect` without a table index; it is unreachable today, so it should
   kpanic. The instantiation-failure path walks `op->next` instead of
   `blockOpCount`.
7. **Docs.** `jitWasmCodeGen.h` refers to `docs/Wasm-JIT-*.md`, which does not
   exist. The real build and test notes are in
   `project/emscripten/buildInstructions.md`.

## Plan (ranked)

1. **Done: audit the other `If { emulateSingleOp } EndIf` + native patterns**
   in `jit/`. There are 16 sites. Memory-access fallbacks (`jitCodeGen.cpp`)
   do not apply because WASM overrides the memory paths. `fwait` always
   redirects control when taken. 8/16-bit div is whole-op fallback on WASM,
   and WASM has its own `div32`/`idiv32` with exact overflow checks. The x87
   `fdiv` guard is covered by a test that passes. `guardSseDiv` was the only
   reachable instance. Any new guard of this shape needs `blockExit()` after
   `emulateSingleOp()`.
2. **Native atomics for MT LOCK ops.** Add `i32.atomic.rmw.{add,sub,and,or,xor,xchg,cmpxchg}`
   (+ `.8_u/.16_u` forms) to `WasmEmitter`, and implement the `*_lock`
   handlers on the inline TLB fast path, keeping the helper for page-crossing
   or unmapped addresses.
3. **Shrink the fallback families** (finding 4), starting with the ones the
   stats build reports as hot. Preserve lazy flags across fallbacks where the
   interpreter does not change them.
4. **Keep the merged-module branch hints** by remapping the custom section by
   function index in `wasmModuleMerger.cpp`.
5. **Robustness fixes** from finding 6.

## Testing in a Linux container

```sh
# Emscripten 5.0.7 is required (BUILD.md)
git clone https://github.com/emscripten-core/emsdk.git /opt/emsdk
/opt/emsdk/emsdk install 5.0.7 && /opt/emsdk/emsdk activate 5.0.7
source /opt/emsdk/emsdk_env.sh
cd project/emscripten && make testJit
cd Build/TestJit && node boxedwine.js            # whole suite
node boxedwine.js -list                          # indexes
node boxedwine.js <start> <count> <threads>      # a range
```

If GitHub archive downloads are blocked, seed the `zlib` and `sdl2` ports from
git checkouts: copy the sources to
`emsdk/upstream/emscripten/cache/ports/<port>/<subdir>`, then write the port
URL into `cache/ports/<port>/.emscripten_url`.
