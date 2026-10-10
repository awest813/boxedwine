/*
 *  Copyright (C) 2012-2025  The BoxedWine Team
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */

#ifndef __RV64_EMITTER_H__
#define __RV64_EMITTER_H__

// Minimal RV64 machine-code emitter for the RISC-V JIT backend. asmjit has no
// RISC-V target, so this covers what the backend needs: RV64I, M, A, Zba and
// Zbb, with labels and branch fixups. Compressed instructions are never
// emitted, so every instruction is one 32-bit word.
//
// The encoder has no dependencies on the rest of Boxedwine so it can be unit
// tested on any host against GNU as (see tools/riscv64/testRv64Emitter.cpp).

#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>

namespace rv64 {

enum Reg : uint8_t {
    zero = 0, ra = 1, sp = 2, gp = 3, tp = 4,
    t0 = 5, t1 = 6, t2 = 7,
    s0 = 8, fp = 8, s1 = 9,
    a0 = 10, a1 = 11, a2 = 12, a3 = 13, a4 = 14, a5 = 15, a6 = 16, a7 = 17,
    s2 = 18, s3 = 19, s4 = 20, s5 = 21, s6 = 22, s7 = 23, s8 = 24, s9 = 25, s10 = 26, s11 = 27,
    t3 = 28, t4 = 29, t5 = 30, t6 = 31
};

enum class Cond : uint8_t {
    EQ = 0, NE = 1, LT = 4, GE = 5, LTU = 6, GEU = 7
};

inline Cond invert(Cond c) {
    return (Cond)((uint8_t)c ^ 1);
}

struct Label {
    uint32_t id = 0xffffffff;
    bool isValid() const { return id != 0xffffffff; }
};

inline bool isInt12(int64_t v) { return v >= -2048 && v <= 2047; }
inline bool isInt32(int64_t v) { return v >= INT32_MIN && v <= INT32_MAX; }

class Emitter {
public:
    // Raised for an out-of-range branch, an unbound label or a bad operand.
    // By default it is thrown; the JIT installs a handler that kpanics.
    class Error : public std::runtime_error {
    public:
        explicit Error(const char* msg) : std::runtime_error(msg) {}
    };
    using ErrorHandler = void (*)(const char* msg);
    static void setErrorHandler(ErrorHandler handler) { errorHandler() = handler; }
    static void fail(const char* msg) {
        if (errorHandler()) {
            errorHandler()(msg);
        }
        throw Error(msg);
    }

    uint32_t size() const { return (uint32_t)(words.size() * 4); }
    const uint8_t* data() const { return (const uint8_t*)words.data(); }
    const std::vector<uint32_t>& code() const { return words; }

    void emit(uint32_t inst) { words.push_back(inst); }

    // ----- labels -----

    Label newLabel() {
        Label l;
        l.id = (uint32_t)labelOffsets.size();
        labelOffsets.push_back(-1);
        return l;
    }

    void bind(Label l) {
        if (labelOffsets.at(l.id) != -1) {
            fail("rv64::Emitter label bound twice");
        }
        labelOffsets[l.id] = (int64_t)size();
    }

    bool isBound(Label l) const { return labelOffsets.at(l.id) != -1; }
    uint32_t labelOffset(Label l) const {
        if (labelOffsets.at(l.id) < 0) {
            fail("rv64::Emitter label not bound");
        }
        return (uint32_t)labelOffsets[l.id];
    }

    // Patch every pending branch. Must be called once after code generation.
    void resolve() {
        for (const Fixup& f : fixups) {
            int64_t target = labelOffsets.at(f.label.id);
            if (target < 0) {
                fail("rv64::Emitter branch to unbound label");
            }
            int64_t delta = target - (int64_t)f.pos;
            uint32_t& inst = words[f.pos / 4];
            if (f.kind == FixupKind::Branch) {
                if (delta < -4096 || delta > 4094) {
                    fail("rv64::Emitter branch out of range");
                }
                inst |= encodeBImm((int32_t)delta);
            } else {
                if (delta < -(1 << 20) || delta > (1 << 20) - 2) {
                    fail("rv64::Emitter jump out of range");
                }
                inst |= encodeJImm((int32_t)delta);
            }
        }
        fixups.clear();
    }

    // ----- raw formats -----

    static uint32_t R(uint32_t opcode, uint32_t f3, uint32_t f7, Reg rd, Reg rs1, Reg rs2) {
        return opcode | ((uint32_t)rd << 7) | (f3 << 12) | ((uint32_t)rs1 << 15) | ((uint32_t)rs2 << 20) | (f7 << 25);
    }
    static uint32_t I(uint32_t opcode, uint32_t f3, Reg rd, Reg rs1, int32_t imm) {
        checkImm12(imm);
        return opcode | ((uint32_t)rd << 7) | (f3 << 12) | ((uint32_t)rs1 << 15) | (((uint32_t)imm & 0xfff) << 20);
    }
    static uint32_t S(uint32_t opcode, uint32_t f3, Reg rs1, Reg rs2, int32_t imm) {
        checkImm12(imm);
        uint32_t u = (uint32_t)imm;
        return opcode | ((u & 0x1f) << 7) | (f3 << 12) | ((uint32_t)rs1 << 15) | ((uint32_t)rs2 << 20) | (((u >> 5) & 0x7f) << 25);
    }
    static uint32_t U(uint32_t opcode, Reg rd, int32_t imm20) {
        return opcode | ((uint32_t)rd << 7) | (((uint32_t)imm20 & 0xfffff) << 12);
    }
    static uint32_t encodeBImm(int32_t off) {
        uint32_t u = (uint32_t)off;
        return (((u >> 12) & 1) << 31) | (((u >> 5) & 0x3f) << 25) | (((u >> 1) & 0xf) << 8) | (((u >> 11) & 1) << 7);
    }
    static uint32_t encodeJImm(int32_t off) {
        uint32_t u = (uint32_t)off;
        return (((u >> 20) & 1) << 31) | (((u >> 1) & 0x3ff) << 21) | (((u >> 11) & 1) << 20) | (((u >> 12) & 0xff) << 12);
    }

    // ----- RV64I -----

    void lui(Reg rd, int32_t imm20) { emit(U(0x37, rd, imm20)); }
    void auipc(Reg rd, int32_t imm20) { emit(U(0x17, rd, imm20)); }

    void jal(Reg rd, int32_t offset) {
        checkAligned(offset);
        if (offset < -(1 << 20) || offset > (1 << 20) - 2) {
            fail("rv64::Emitter jal offset out of range");
        }
        emit(0x6f | ((uint32_t)rd << 7) | encodeJImm(offset));
    }
    void jal(Reg rd, Label l) {
        fixups.push_back({ size(), l, FixupKind::Jump });
        emit(0x6f | ((uint32_t)rd << 7));
    }
    void jalr(Reg rd, Reg rs1, int32_t imm = 0) { emit(I(0x67, 0, rd, rs1, imm)); }

    void branch(Cond c, Reg rs1, Reg rs2, int32_t offset) {
        checkAligned(offset);
        if (offset < -4096 || offset > 4094) {
            fail("rv64::Emitter branch offset out of range");
        }
        emit(0x63 | ((uint32_t)c << 12) | ((uint32_t)rs1 << 15) | ((uint32_t)rs2 << 20) | encodeBImm(offset));
    }
    // Short form: the label must end up within +-4 KiB.
    void branch(Cond c, Reg rs1, Reg rs2, Label l) {
        fixups.push_back({ size(), l, FixupKind::Branch });
        emit(0x63 | ((uint32_t)c << 12) | ((uint32_t)rs1 << 15) | ((uint32_t)rs2 << 20));
    }
    // Long form: reaches +-1 MiB by branching over a jal with the inverted
    // condition. Used for structured If/EndIf where the body size is unknown.
    void branchFar(Cond c, Reg rs1, Reg rs2, Label l) {
        branch(invert(c), rs1, rs2, 8);
        jal(zero, l);
    }

    void beq(Reg a, Reg b, int32_t o) { branch(Cond::EQ, a, b, o); }
    void bne(Reg a, Reg b, int32_t o) { branch(Cond::NE, a, b, o); }
    void blt(Reg a, Reg b, int32_t o) { branch(Cond::LT, a, b, o); }
    void bge(Reg a, Reg b, int32_t o) { branch(Cond::GE, a, b, o); }
    void bltu(Reg a, Reg b, int32_t o) { branch(Cond::LTU, a, b, o); }
    void bgeu(Reg a, Reg b, int32_t o) { branch(Cond::GEU, a, b, o); }

    void lb(Reg rd, Reg rs1, int32_t imm) { emit(I(0x03, 0, rd, rs1, imm)); }
    void lh(Reg rd, Reg rs1, int32_t imm) { emit(I(0x03, 1, rd, rs1, imm)); }
    void lw(Reg rd, Reg rs1, int32_t imm) { emit(I(0x03, 2, rd, rs1, imm)); }
    void ld(Reg rd, Reg rs1, int32_t imm) { emit(I(0x03, 3, rd, rs1, imm)); }
    void lbu(Reg rd, Reg rs1, int32_t imm) { emit(I(0x03, 4, rd, rs1, imm)); }
    void lhu(Reg rd, Reg rs1, int32_t imm) { emit(I(0x03, 5, rd, rs1, imm)); }
    void lwu(Reg rd, Reg rs1, int32_t imm) { emit(I(0x03, 6, rd, rs1, imm)); }

    void sb(Reg rs2, Reg rs1, int32_t imm) { emit(S(0x23, 0, rs1, rs2, imm)); }
    void sh(Reg rs2, Reg rs1, int32_t imm) { emit(S(0x23, 1, rs1, rs2, imm)); }
    void sw(Reg rs2, Reg rs1, int32_t imm) { emit(S(0x23, 2, rs1, rs2, imm)); }
    void sd(Reg rs2, Reg rs1, int32_t imm) { emit(S(0x23, 3, rs1, rs2, imm)); }

    void addi(Reg rd, Reg rs1, int32_t imm) { emit(I(0x13, 0, rd, rs1, imm)); }
    void slti(Reg rd, Reg rs1, int32_t imm) { emit(I(0x13, 2, rd, rs1, imm)); }
    void sltiu(Reg rd, Reg rs1, int32_t imm) { emit(I(0x13, 3, rd, rs1, imm)); }
    void xori(Reg rd, Reg rs1, int32_t imm) { emit(I(0x13, 4, rd, rs1, imm)); }
    void ori(Reg rd, Reg rs1, int32_t imm) { emit(I(0x13, 6, rd, rs1, imm)); }
    void andi(Reg rd, Reg rs1, int32_t imm) { emit(I(0x13, 7, rd, rs1, imm)); }
    void slli(Reg rd, Reg rs1, uint32_t sh) { checkShift(sh, 63); emit(I(0x13, 1, rd, rs1, (int32_t)sh)); }
    void srli(Reg rd, Reg rs1, uint32_t sh) { checkShift(sh, 63); emit(I(0x13, 5, rd, rs1, (int32_t)sh)); }
    void srai(Reg rd, Reg rs1, uint32_t sh) { checkShift(sh, 63); emit(I(0x13, 5, rd, rs1, (int32_t)(sh | 0x400))); }

    void add(Reg rd, Reg a, Reg b) { emit(R(0x33, 0, 0x00, rd, a, b)); }
    void sub(Reg rd, Reg a, Reg b) { emit(R(0x33, 0, 0x20, rd, a, b)); }
    void sll(Reg rd, Reg a, Reg b) { emit(R(0x33, 1, 0x00, rd, a, b)); }
    void slt(Reg rd, Reg a, Reg b) { emit(R(0x33, 2, 0x00, rd, a, b)); }
    void sltu(Reg rd, Reg a, Reg b) { emit(R(0x33, 3, 0x00, rd, a, b)); }
    void xor_(Reg rd, Reg a, Reg b) { emit(R(0x33, 4, 0x00, rd, a, b)); }
    void srl(Reg rd, Reg a, Reg b) { emit(R(0x33, 5, 0x00, rd, a, b)); }
    void sra(Reg rd, Reg a, Reg b) { emit(R(0x33, 5, 0x20, rd, a, b)); }
    void or_(Reg rd, Reg a, Reg b) { emit(R(0x33, 6, 0x00, rd, a, b)); }
    void and_(Reg rd, Reg a, Reg b) { emit(R(0x33, 7, 0x00, rd, a, b)); }

    void addiw(Reg rd, Reg rs1, int32_t imm) { emit(I(0x1b, 0, rd, rs1, imm)); }
    void slliw(Reg rd, Reg rs1, uint32_t sh) { checkShift(sh, 31); emit(I(0x1b, 1, rd, rs1, (int32_t)sh)); }
    void srliw(Reg rd, Reg rs1, uint32_t sh) { checkShift(sh, 31); emit(I(0x1b, 5, rd, rs1, (int32_t)sh)); }
    void sraiw(Reg rd, Reg rs1, uint32_t sh) { checkShift(sh, 31); emit(I(0x1b, 5, rd, rs1, (int32_t)(sh | 0x400))); }
    void addw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 0, 0x00, rd, a, b)); }
    void subw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 0, 0x20, rd, a, b)); }
    void sllw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 1, 0x00, rd, a, b)); }
    void srlw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 5, 0x00, rd, a, b)); }
    void sraw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 5, 0x20, rd, a, b)); }

    // fence rw,rw
    void fence() { emit(0x0330000f); }
    void fence_i() { emit(0x0000100f); }
    void ebreak() { emit(0x00100073); }
    // Defined-illegal instruction (all zeros traps on every implementation).
    void illegal() { emit(0); }

    // ----- M -----

    void mul(Reg rd, Reg a, Reg b) { emit(R(0x33, 0, 0x01, rd, a, b)); }
    void mulh(Reg rd, Reg a, Reg b) { emit(R(0x33, 1, 0x01, rd, a, b)); }
    void mulhsu(Reg rd, Reg a, Reg b) { emit(R(0x33, 2, 0x01, rd, a, b)); }
    void mulhu(Reg rd, Reg a, Reg b) { emit(R(0x33, 3, 0x01, rd, a, b)); }
    void div(Reg rd, Reg a, Reg b) { emit(R(0x33, 4, 0x01, rd, a, b)); }
    void divu(Reg rd, Reg a, Reg b) { emit(R(0x33, 5, 0x01, rd, a, b)); }
    void rem(Reg rd, Reg a, Reg b) { emit(R(0x33, 6, 0x01, rd, a, b)); }
    void remu(Reg rd, Reg a, Reg b) { emit(R(0x33, 7, 0x01, rd, a, b)); }
    void mulw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 0, 0x01, rd, a, b)); }
    void divw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 4, 0x01, rd, a, b)); }
    void divuw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 5, 0x01, rd, a, b)); }
    void remw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 6, 0x01, rd, a, b)); }
    void remuw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 7, 0x01, rd, a, b)); }

    // ----- A (always .aqrl: these implement x86 LOCK semantics) -----

    void lr_w(Reg rd, Reg addr) { emit(R(0x2f, 2, (0x02 << 2) | 3, rd, addr, zero)); }
    void lr_d(Reg rd, Reg addr) { emit(R(0x2f, 3, (0x02 << 2) | 3, rd, addr, zero)); }
    void sc_w(Reg rd, Reg value, Reg addr) { emit(R(0x2f, 2, (0x03 << 2) | 3, rd, addr, value)); }
    void sc_d(Reg rd, Reg value, Reg addr) { emit(R(0x2f, 3, (0x03 << 2) | 3, rd, addr, value)); }
    void amoswap_w(Reg rd, Reg value, Reg addr) { emit(R(0x2f, 2, (0x01 << 2) | 3, rd, addr, value)); }
    void amoadd_w(Reg rd, Reg value, Reg addr) { emit(R(0x2f, 2, (0x00 << 2) | 3, rd, addr, value)); }
    void amoxor_w(Reg rd, Reg value, Reg addr) { emit(R(0x2f, 2, (0x04 << 2) | 3, rd, addr, value)); }
    void amoand_w(Reg rd, Reg value, Reg addr) { emit(R(0x2f, 2, (0x0c << 2) | 3, rd, addr, value)); }
    void amoor_w(Reg rd, Reg value, Reg addr) { emit(R(0x2f, 2, (0x08 << 2) | 3, rd, addr, value)); }

    // ----- Zba -----

    void add_uw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 0, 0x04, rd, a, b)); }
    void sh1add(Reg rd, Reg a, Reg b) { emit(R(0x33, 2, 0x10, rd, a, b)); }
    void sh2add(Reg rd, Reg a, Reg b) { emit(R(0x33, 4, 0x10, rd, a, b)); }
    void sh3add(Reg rd, Reg a, Reg b) { emit(R(0x33, 6, 0x10, rd, a, b)); }
    void sh1add_uw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 2, 0x10, rd, a, b)); }
    void sh2add_uw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 4, 0x10, rd, a, b)); }
    void sh3add_uw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 6, 0x10, rd, a, b)); }
    void slli_uw(Reg rd, Reg rs1, uint32_t sh) { checkShift(sh, 63); emit(I(0x1b, 1, rd, rs1, (int32_t)(sh | 0x080))); }
    void zext_w(Reg rd, Reg rs) { add_uw(rd, rs, zero); }

    // ----- Zbb -----

    void andn(Reg rd, Reg a, Reg b) { emit(R(0x33, 7, 0x20, rd, a, b)); }
    void orn(Reg rd, Reg a, Reg b) { emit(R(0x33, 6, 0x20, rd, a, b)); }
    void xnor(Reg rd, Reg a, Reg b) { emit(R(0x33, 4, 0x20, rd, a, b)); }
    void clz(Reg rd, Reg rs) { emit(I(0x13, 1, rd, rs, 0x600)); }
    void ctz(Reg rd, Reg rs) { emit(I(0x13, 1, rd, rs, 0x601)); }
    void cpop(Reg rd, Reg rs) { emit(I(0x13, 1, rd, rs, 0x602)); }
    void clzw(Reg rd, Reg rs) { emit(I(0x1b, 1, rd, rs, 0x600)); }
    void ctzw(Reg rd, Reg rs) { emit(I(0x1b, 1, rd, rs, 0x601)); }
    void cpopw(Reg rd, Reg rs) { emit(I(0x1b, 1, rd, rs, 0x602)); }
    void sext_b(Reg rd, Reg rs) { emit(I(0x13, 1, rd, rs, 0x604)); }
    void sext_h(Reg rd, Reg rs) { emit(I(0x13, 1, rd, rs, 0x605)); }
    void zext_h(Reg rd, Reg rs) { emit(R(0x3b, 4, 0x04, rd, rs, zero)); }
    void max(Reg rd, Reg a, Reg b) { emit(R(0x33, 6, 0x05, rd, a, b)); }
    void maxu(Reg rd, Reg a, Reg b) { emit(R(0x33, 7, 0x05, rd, a, b)); }
    void min(Reg rd, Reg a, Reg b) { emit(R(0x33, 4, 0x05, rd, a, b)); }
    void minu(Reg rd, Reg a, Reg b) { emit(R(0x33, 5, 0x05, rd, a, b)); }
    void rol(Reg rd, Reg a, Reg b) { emit(R(0x33, 1, 0x30, rd, a, b)); }
    void ror(Reg rd, Reg a, Reg b) { emit(R(0x33, 5, 0x30, rd, a, b)); }
    void rolw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 1, 0x30, rd, a, b)); }
    void rorw(Reg rd, Reg a, Reg b) { emit(R(0x3b, 5, 0x30, rd, a, b)); }
    void rori(Reg rd, Reg rs, uint32_t sh) { checkShift(sh, 63); emit(I(0x13, 5, rd, rs, (int32_t)(0x600 | sh))); }
    void roriw(Reg rd, Reg rs, uint32_t sh) { checkShift(sh, 31); emit(I(0x1b, 5, rd, rs, (int32_t)(0x600 | sh))); }
    void rev8(Reg rd, Reg rs) { emit(I(0x13, 5, rd, rs, 0x6b8)); }
    void orc_b(Reg rd, Reg rs) { emit(I(0x13, 5, rd, rs, 0x287)); }

    // ----- pseudo instructions -----

    void nop() { addi(zero, zero, 0); }
    void mv(Reg rd, Reg rs) { if (rd != rs) addi(rd, rs, 0); }
    void not_(Reg rd, Reg rs) { xori(rd, rs, -1); }
    void neg(Reg rd, Reg rs) { sub(rd, zero, rs); }
    void negw(Reg rd, Reg rs) { subw(rd, zero, rs); }
    void sext_w(Reg rd, Reg rs) { addiw(rd, rs, 0); }
    void seqz(Reg rd, Reg rs) { sltiu(rd, rs, 1); }
    void snez(Reg rd, Reg rs) { sltu(rd, zero, rs); }
    void sltz(Reg rd, Reg rs) { slt(rd, rs, zero); }
    void sgtz(Reg rd, Reg rs) { slt(rd, zero, rs); }
    void j(Label l) { jal(zero, l); }
    void jr(Reg rs) { jalr(zero, rs, 0); }
    void jalr(Reg rs) { jalr(ra, rs, 0); }
    void ret() { jalr(zero, ra, 0); }

    // Load an arbitrary 64-bit constant (at most 8 instructions, usually 1-3).
    void li(Reg rd, int64_t value) {
        if (isInt32(value)) {
            int32_t lo = signExtend12((uint32_t)value);
            int32_t hi = (int32_t)(((int64_t)value - lo) >> 12);
            if (hi) {
                lui(rd, hi);
                if (lo) {
                    addiw(rd, rd, lo);
                }
            } else {
                addi(rd, zero, lo);
            }
            return;
        }
        int32_t lo = signExtend12((uint32_t)value);
        int64_t hi = (int64_t)(((uint64_t)value - (uint64_t)(int64_t)lo)) >> 12;
        uint32_t shift = 12;
        while (!(hi & 1)) {
            hi >>= 1;
            shift++;
        }
        li(rd, hi);
        slli(rd, rd, shift);
        if (lo) {
            addi(rd, rd, lo);
        }
    }

    static int32_t signExtend12(uint32_t v) {
        return (int32_t)(v << 20) >> 20;
    }

private:
    enum class FixupKind : uint8_t { Branch, Jump };
    struct Fixup {
        uint32_t pos;
        Label label;
        FixupKind kind;
    };

    static void checkImm12(int32_t imm) {
        if (!isInt12(imm)) {
            fail("rv64::Emitter immediate does not fit in 12 bits");
        }
    }
    static void checkShift(uint32_t sh, uint32_t max) {
        if (sh > max) {
            fail("rv64::Emitter shift amount out of range");
        }
    }
    static void checkAligned(int32_t offset) {
        if (offset & 1) {
            fail("rv64::Emitter misaligned branch offset");
        }
    }

    static ErrorHandler& errorHandler() {
        static ErrorHandler handler = nullptr;
        return handler;
    }

    std::vector<uint32_t> words;
    std::vector<int64_t> labelOffsets;
    std::vector<Fixup> fixups;
};

} // namespace rv64

#endif
