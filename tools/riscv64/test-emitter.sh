#!/bin/sh
# Verifies the RISC-V JIT encoder against GNU as, then executes generated code
# under qemu-riscv64. Needs g++, riscv64-linux-gnu-{g++,as,objcopy}, qemu-riscv64.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${TMPDIR:-/tmp}/rv64-emitter-test
mkdir -p "$OUT"
SRC="$ROOT/tools/riscv64/testRv64Emitter.cpp"
INC="-I$ROOT/source/emulation/cpu/riscv64"

g++ -std=c++20 -O1 -Wall $INC "$SRC" -o "$OUT/encode"
"$OUT/encode" encode "$OUT/emitter.s" "$OUT/emitter.bin"
riscv64-linux-gnu-as -march=rv64gc_zba_zbb "$OUT/emitter.s" -o "$OUT/emitter.o"
riscv64-linux-gnu-objcopy -O binary -j .text "$OUT/emitter.o" "$OUT/as.bin"
if cmp "$OUT/emitter.bin" "$OUT/as.bin"; then
    echo "encoding: OK"
else
    echo "encoding: MISMATCH (compare with riscv64-linux-gnu-objdump -D -b binary -m riscv:rv64)"
    exit 1
fi

riscv64-linux-gnu-g++ -std=c++20 -O1 -static $INC "$SRC" -o "$OUT/exec"
qemu-riscv64 -cpu rv64,zba=true,zbb=true "$OUT/exec" exec
