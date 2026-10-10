// Unit test for source/emulation/cpu/riscv64/rv64Emitter.h
//
// Encoding check (any host):
//   g++ -std=c++20 -I source/emulation/cpu/riscv64 tools/riscv64/testRv64Emitter.cpp -o /tmp/rv64test
//   /tmp/rv64test encode /tmp/rv64.s /tmp/rv64.bin
//   riscv64-linux-gnu-as -march=rv64gc_zba_zbb /tmp/rv64.s -o /tmp/rv64.o
//   riscv64-linux-gnu-objcopy -O binary -j .text /tmp/rv64.o /tmp/rv64-as.bin
//   cmp /tmp/rv64.bin /tmp/rv64-as.bin
// tools/riscv64/test-emitter.sh runs all of the above.
//
// Execution check (riscv64 host or qemu-riscv64): runs generated li/ALU code.
//   riscv64-linux-gnu-g++ -std=c++20 -static -I source/emulation/cpu/riscv64 tools/riscv64/testRv64Emitter.cpp -o /tmp/rv64test-rv
//   qemu-riscv64 /tmp/rv64test-rv exec

#include "rv64Emitter.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>
#include <cinttypes>
#if defined(__riscv)
#include <sys/mman.h>
#endif

using namespace rv64;

struct Case {
    std::string text;
    std::function<void(Emitter&)> emit;
};

static const char* regName(Reg r) {
    static const char* names[] = { "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1",
        "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9",
        "s10", "s11", "t3", "t4", "t5", "t6" };
    return names[r];
}

static std::vector<Case> buildCases() {
    std::vector<Case> c;
    auto add = [&c](std::string text, std::function<void(Emitter&)> fn) { c.push_back({ std::move(text), std::move(fn) }); };
    const Reg regs[] = { zero, ra, a0, a5, s1, s11, t6, t0 };
    char buf[128];

    // R-type ops over a spread of registers
    struct ROp { const char* name; void (Emitter::*fn)(Reg, Reg, Reg); };
    const ROp rops[] = {
        {"add", &Emitter::add}, {"sub", &Emitter::sub}, {"sll", &Emitter::sll}, {"slt", &Emitter::slt},
        {"sltu", &Emitter::sltu}, {"xor", &Emitter::xor_}, {"srl", &Emitter::srl}, {"sra", &Emitter::sra},
        {"or", &Emitter::or_}, {"and", &Emitter::and_}, {"addw", &Emitter::addw}, {"subw", &Emitter::subw},
        {"sllw", &Emitter::sllw}, {"srlw", &Emitter::srlw}, {"sraw", &Emitter::sraw},
        {"mul", &Emitter::mul}, {"mulh", &Emitter::mulh}, {"mulhsu", &Emitter::mulhsu}, {"mulhu", &Emitter::mulhu},
        {"div", &Emitter::div}, {"divu", &Emitter::divu}, {"rem", &Emitter::rem}, {"remu", &Emitter::remu},
        {"mulw", &Emitter::mulw}, {"divw", &Emitter::divw}, {"divuw", &Emitter::divuw}, {"remw", &Emitter::remw}, {"remuw", &Emitter::remuw},
        {"add.uw", &Emitter::add_uw}, {"sh1add", &Emitter::sh1add}, {"sh2add", &Emitter::sh2add}, {"sh3add", &Emitter::sh3add},
        {"sh1add.uw", &Emitter::sh1add_uw}, {"sh2add.uw", &Emitter::sh2add_uw}, {"sh3add.uw", &Emitter::sh3add_uw},
        {"andn", &Emitter::andn}, {"orn", &Emitter::orn}, {"xnor", &Emitter::xnor},
        {"max", &Emitter::max}, {"maxu", &Emitter::maxu}, {"min", &Emitter::min}, {"minu", &Emitter::minu},
        {"rol", &Emitter::rol}, {"ror", &Emitter::ror}, {"rolw", &Emitter::rolw}, {"rorw", &Emitter::rorw},
    };
    for (const ROp& op : rops) {
        for (int i = 0; i < 3; i++) {
            Reg d = regs[(i * 3 + 1) % 8], a = regs[(i * 5 + 2) % 8], b = regs[(i * 7 + 3) % 8];
            snprintf(buf, sizeof(buf), "%s %s, %s, %s", op.name, regName(d), regName(a), regName(b));
            auto fn = op.fn;
            add(buf, [fn, d, a, b](Emitter& e) { (e.*fn)(d, a, b); });
        }
    }

    // I-type with immediates covering the edges
    struct IOp { const char* name; void (Emitter::*fn)(Reg, Reg, int32_t); };
    const IOp iops[] = {
        {"addi", &Emitter::addi}, {"slti", &Emitter::slti}, {"sltiu", &Emitter::sltiu}, {"xori", &Emitter::xori},
        {"ori", &Emitter::ori}, {"andi", &Emitter::andi}, {"addiw", &Emitter::addiw},
    };
    const int32_t imms[] = { 0, 1, -1, 2047, -2048, 0x123, -0x456 };
    for (const IOp& op : iops) {
        for (int32_t imm : imms) {
            snprintf(buf, sizeof(buf), "%s a0, s2, %d", op.name, imm);
            auto fn = op.fn;
            add(buf, [fn, imm](Emitter& e) { (e.*fn)(a0, s2, imm); });
        }
    }

    // shifts
    struct SOp { const char* name; void (Emitter::*fn)(Reg, Reg, uint32_t); uint32_t max; };
    const SOp sops[] = {
        {"slli", &Emitter::slli, 63}, {"srli", &Emitter::srli, 63}, {"srai", &Emitter::srai, 63},
        {"slliw", &Emitter::slliw, 31}, {"srliw", &Emitter::srliw, 31}, {"sraiw", &Emitter::sraiw, 31},
        {"rori", &Emitter::rori, 63}, {"roriw", &Emitter::roriw, 31}, {"slli.uw", &Emitter::slli_uw, 63},
    };
    for (const SOp& op : sops) {
        for (uint32_t sh : { 0u, 1u, 13u, op.max }) {
            snprintf(buf, sizeof(buf), "%s t1, a7, %u", op.name, sh);
            auto fn = op.fn;
            add(buf, [fn, sh](Emitter& e) { (e.*fn)(t1, a7, sh); });
        }
    }

    // loads / stores
    struct LOp { const char* name; void (Emitter::*fn)(Reg, Reg, int32_t); };
    const LOp lops[] = {
        {"lb", &Emitter::lb}, {"lh", &Emitter::lh}, {"lw", &Emitter::lw}, {"ld", &Emitter::ld},
        {"lbu", &Emitter::lbu}, {"lhu", &Emitter::lhu}, {"lwu", &Emitter::lwu},
        {"sb", &Emitter::sb}, {"sh", &Emitter::sh}, {"sw", &Emitter::sw}, {"sd", &Emitter::sd},
    };
    for (const LOp& op : lops) {
        for (int32_t imm : { 0, 8, -8, 2047, -2048 }) {
            snprintf(buf, sizeof(buf), "%s a3, %d(s1)", op.name, imm);
            auto fn = op.fn;
            add(buf, [fn, imm](Emitter& e) { (e.*fn)(a3, s1, imm); });
        }
    }

    // unary Zbb
    struct UOp { const char* name; void (Emitter::*fn)(Reg, Reg); };
    const UOp uops[] = {
        {"clz", &Emitter::clz}, {"ctz", &Emitter::ctz}, {"cpop", &Emitter::cpop}, {"clzw", &Emitter::clzw},
        {"ctzw", &Emitter::ctzw}, {"cpopw", &Emitter::cpopw}, {"sext.b", &Emitter::sext_b}, {"sext.h", &Emitter::sext_h},
        {"zext.h", &Emitter::zext_h}, {"rev8", &Emitter::rev8}, {"orc.b", &Emitter::orc_b}, {"zext.w", &Emitter::zext_w},
    };
    for (const UOp& op : uops) {
        snprintf(buf, sizeof(buf), "%s s4, t3", op.name);
        auto fn = op.fn;
        add(buf, [fn](Emitter& e) { (e.*fn)(s4, t3); });
    }

    // atomics
    add("lr.w.aqrl a0, (a1)", [](Emitter& e) { e.lr_w(a0, a1); });
    add("lr.d.aqrl a0, (a1)", [](Emitter& e) { e.lr_d(a0, a1); });
    add("sc.w.aqrl a2, a3, (a1)", [](Emitter& e) { e.sc_w(a2, a3, a1); });
    add("sc.d.aqrl a2, a3, (a1)", [](Emitter& e) { e.sc_d(a2, a3, a1); });
    add("amoswap.w.aqrl a2, a3, (a1)", [](Emitter& e) { e.amoswap_w(a2, a3, a1); });
    add("amoadd.w.aqrl a2, a3, (a1)", [](Emitter& e) { e.amoadd_w(a2, a3, a1); });
    add("amoxor.w.aqrl a2, a3, (a1)", [](Emitter& e) { e.amoxor_w(a2, a3, a1); });
    add("amoand.w.aqrl a2, a3, (a1)", [](Emitter& e) { e.amoand_w(a2, a3, a1); });
    add("amoor.w.aqrl a2, a3, (a1)", [](Emitter& e) { e.amoor_w(a2, a3, a1); });

    // misc
    add("lui t2, 0xfffff", [](Emitter& e) { e.lui(t2, 0xfffff); });
    add("lui t2, 0x12345", [](Emitter& e) { e.lui(t2, 0x12345); });
    add("auipc t2, 0x1", [](Emitter& e) { e.auipc(t2, 1); });
    add("jalr ra, 16(a4)", [](Emitter& e) { e.jalr(ra, a4, 16); });
    add("jalr zero, 0(t0)", [](Emitter& e) { e.jr(t0); });
    add("ret", [](Emitter& e) { e.ret(); });
    add("fence rw, rw", [](Emitter& e) { e.fence(); });
    add("fence.i", [](Emitter& e) { e.fence_i(); });
    add("ebreak", [](Emitter& e) { e.ebreak(); });

    // branches and jumps with immediate offsets (GNU as resolves "." relative)
    add("beq a0, a1, .+8", [](Emitter& e) { e.beq(a0, a1, 8); });
    add("bne a0, a1, .-8", [](Emitter& e) { e.bne(a0, a1, -8); });
    add("blt a0, a1, .+4094", [](Emitter& e) { e.blt(a0, a1, 4094); });
    add("bge a0, a1, .-4096", [](Emitter& e) { e.bge(a0, a1, -4096); });
    add("bltu s3, t4, .+256", [](Emitter& e) { e.bltu(s3, t4, 256); });
    add("bgeu s3, t4, .-2", [](Emitter& e) { e.bgeu(s3, t4, -2); });
    add("jal zero, .+1048574", [](Emitter& e) { e.jal(zero, 1048574); });
    add("jal ra, .-1048576", [](Emitter& e) { e.jal(ra, -1048576); });
    add("jal ra, .+2048", [](Emitter& e) { e.jal(ra, 2048); });
    return c;
}

static int encode(const char* asmPath, const char* binPath) {
    std::vector<Case> cases = buildCases();
    FILE* s = fopen(asmPath, "w");
    FILE* b = fopen(binPath, "wb");
    if (!s || !b) {
        perror("open");
        return 1;
    }
    fprintf(s, ".option norvc\n.option norelax\n.text\n");
    for (const Case& c : cases) {
        Emitter e;
        c.emit(e);
        e.resolve();
        if (e.size() != 4) {
            fprintf(stderr, "case '%s' emitted %u bytes\n", c.text.c_str(), e.size());
            return 1;
        }
        fprintf(s, "%s\n", c.text.c_str());
        fwrite(e.data(), 1, 4, b);
    }
    // Labels: forward and backward short branches, a far branch and a jal.
    {
        Emitter e;
        Label back = e.newLabel();
        Label fwd = e.newLabel();
        e.bind(back);
        e.branch(Cond::NE, a0, a1, fwd);
        e.nop();
        e.branch(Cond::LTU, a0, a1, back);
        e.branchFar(Cond::GE, a2, zero, fwd);
        e.j(back);
        e.bind(fwd);
        e.nop();
        e.resolve();
        fprintf(s, "1:\nbne a0, a1, 2f\naddi zero, zero, 0\nbltu a0, a1, 1b\nblt a2, zero, .+8\njal zero, 2f\njal zero, 1b\n2:\naddi zero, zero, 0\n");
        fwrite(e.data(), 1, e.size(), b);
    }
    fclose(s);
    fclose(b);
    printf("wrote %zu single-instruction cases + label case\n", cases.size());
    return 0;
}

#if defined(__riscv)
static int execTests() {
    std::vector<int64_t> values = {
        0, 1, -1, 2047, -2048, 2048, -2049, 0x7ff, 0x800, 0xfff, 0x1000, 0x12345678, 0x7fffffff, -0x80000000LL,
        0x7ffff800, 0x7ffff7ff, 0x80000000LL, 0xffffffffLL, 0x100000000LL, 0xfffff800LL, 0x123456789abcdef0LL,
        (int64_t)0xfedcba9876543210ULL, INT64_MAX, INT64_MIN, 0x0000800000000000LL, (int64_t)0x8000000000000001ULL,
        0x00000000fffff000LL, 0x0000ffffffffffffLL, (int64_t)0xffff0000ffff0000ULL, 0x5555555555555555LL,
    };
    for (int i = 0; i < 2000; i++) {
        uint64_t r = ((uint64_t)rand() << 33) ^ ((uint64_t)rand() << 11) ^ (uint64_t)rand();
        values.push_back((int64_t)r);
        values.push_back((int64_t)(uint32_t)r);
        values.push_back((int64_t)(int32_t)r);
    }
    void* mem = mmap(nullptr, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    int failures = 0;
    size_t maxLen = 0;
    for (int64_t v : values) {
        Emitter e;
        e.li(a0, v);
        e.ret();
        e.resolve();
        maxLen = e.size() > maxLen ? e.size() : maxLen;
        memcpy(mem, e.data(), e.size());
        __builtin___clear_cache((char*)mem, (char*)mem + e.size());
        int64_t got = ((int64_t(*)())mem)();
        if (got != v) {
            printf("li FAILED: expected 0x%016" PRIx64 " got 0x%016" PRIx64 "\n", (uint64_t)v, (uint64_t)got);
            failures++;
        }
    }
    printf("li: %zu values, longest sequence %zu instructions\n", values.size(), maxLen / 4 - 1);

    // Far branches across a gap larger than the short-branch range.
    {
        Emitter e;
        Label skip = e.newLabel();
        e.li(a0, 1);
        e.branchFar(Cond::EQ, zero, zero, skip);
        for (int i = 0; i < 2000; i++) e.li(a0, 99); // 8000 bytes
        e.bind(skip);
        e.ret();
        e.resolve();
        void* big = mmap(nullptr, e.size(), PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        memcpy(big, e.data(), e.size());
        __builtin___clear_cache((char*)big, (char*)big + e.size());
        if (((int64_t(*)())big)() != 1) {
            printf("branchFar FAILED\n");
            failures++;
        }
    }
    printf("%d failures\n", failures);
    return failures ? 1 : 0;
}
#endif

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "encode") {
        try {
            return encode(argv[2], argv[3]);
        } catch (const Emitter::Error& e) {
            fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
    }
#if defined(__riscv)
    if (argc == 2 && std::string(argv[1]) == "exec") {
        return execTests();
    }
#endif
    fprintf(stderr, "usage: %s encode out.s out.bin | exec (riscv64 only)\n", argv[0]);
    return 2;
}
