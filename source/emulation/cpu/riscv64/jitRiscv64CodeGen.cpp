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

#include "boxedwine.h"
#if defined(BOXEDWINE_JIT_RISCV64)

// RISC-V 64 JIT backend (RV64GC + Zba + Zbb). See docs/RISCV64-JIT-Plan.md.
//
// Conventions:
//  * A host register holding a 32-bit or narrower guest value is always zero
//    extended to 64 bits. RISC-V *W instructions sign extend, so 32-bit
//    results are normalized with zext.w.
//  * 8/16-bit operations compute into a scratch register and insert the low
//    bits into the destination, preserving the rest, the same as the ARMv8
//    backend's bfi/bfxil sequences.
//  * Primitives use only the two scratch registers (SCR0/SCR1), never
//    getTmpReg(), so the lazy-flag CF helpers (built with disableTmps) can use
//    them. Ops that need more registers (rcl/rcr, div, ...) are never reached
//    from those helpers.
//  * SSE, MMX and x87 are not implemented yet; the Jit base class sends them
//    through emulateSingleOp.

#include "jitRiscv64CodeGen.h"
#include "../jit/jitCodeGen.h"
#include "../../softmmu/kmemory_soft.h"
#include "rv64Emitter.h"
#include <array>

using namespace rv64;
// Boxedwine has its own global Reg (a guest register), so name the host one HReg.
using HReg = rv64::Reg;

#define NUMBER_OF_REGS 32
#define INVALID_REG 0xff

// Guest EAX..EDI live in callee-saved registers, so they survive host calls.
static const HReg guestRegs[8] = { s2, s3, s4, s5, s6, s7, s8, s9 };
// The last three temps double as the condition-calculation registers used by
// the naked-called CF helpers.
static const HReg tmpRegs[] = { t0, t1, t2, t3, t4, a2, a3, a4, s0, a5, a6, a7 };
#define NUMBER_OF_TMPS ((int)(sizeof(tmpRegs) / sizeof(tmpRegs[0])))

static const HReg regCPU = s1;
static const HReg regMMU = s10;
static const HReg regEmulateSingleOp = s11;
static const HReg SCR0 = t6;
static const HReg SCR1 = t5;

static bool isGuestHostReg(U8 reg) {
    return reg >= s2 && reg <= s9;
}

static bool isCallerSaved(U8 reg) {
    return (reg >= t0 && reg <= t2) || (reg >= a0 && reg <= a7) || reg >= t3;
}

class JitRiscv64CodeGen : public JitCodeGen {
public:
    JitRiscv64CodeGen(CPU* cpu) : JitCodeGen(cpu) {}

    void preOp(DecodedOp* op) override;
    void emulateSingleOp() override;

    // registers
    RegPtr getReg(U8 reg, S8 hint = -1, bool load = true) override;
    RegPtr getReg8(U8 reg, bool load = true) override;
    RegPtr getReadOnlyReg(U8 reg, bool delayed = false, S8 hint = -1) override;
    RegPtr getReadOnlyReg8(U8 reg, bool delayed = false, S8 hint = -1) override;
    RegPtr getTmpReg() override;
    RegPtr getTmpReg8() override;
    RegPtr getTmpRegWithHint(S8 hint) override;
    RegPtr getTmpRegForCallResult() override;
    RegPtr getTmpReg(U8 reg, bool delayed = false, S8 hint = -1) override;
    RegPtr getTmpReg8(U8 reg, bool delayed = false, S8 hint = -1) override;
    RegPtr getReadOnlySegAddress(U8 reg) override;
    RegPtr getTmpSegAddress(U8 reg) override;
    RegPtr getReadOnlySegValue(U8 reg) override;
    RegPtr readEip() override;
    void writeEip(RegPtr eip) override;
    void writeEip(U32 eip) override;
    U32 getAvailableTmpRegCount() override;
    bool isTmpRegAvailable() override;
    void forceSyncBackIfNotCached(RegPtr reg) override {}
    RegPtr getConditionCalculationReg(U32 index) override;
    void fixupU32CallResult(RegPtr reg) override;

    // arithmetic
    void addReg(JitWidth w, RegPtr reg, RegPtr rm) override;
    void addRegWithDest(JitWidth w, RegPtr dst, RegPtr reg, RegPtr rm) override;
    void addValue(JitWidth w, RegPtr reg, U32 imm) override;
    void addValueWithDest(JitWidth w, RegPtr dst, RegPtr reg, U32 imm) override;
    void orReg(JitWidth w, RegPtr reg, RegPtr rm) override;
    void orValue(JitWidth w, RegPtr reg, U32 imm) override;
    void subReg(JitWidth w, RegPtr reg, RegPtr rm) override;
    void subValue(JitWidth w, RegPtr reg, U32 imm) override;
    void subValueWithDest(JitWidth w, RegPtr dst, RegPtr reg, U32 imm) override;
    void andReg(JitWidth w, RegPtr reg, RegPtr rm) override;
    void andValue(JitWidth w, RegPtr reg, U32 imm) override;
    void andValue64(RegPtr reg, U64 imm) override;
    void andValueWithDest(JitWidth w, RegPtr dst, RegPtr reg, U32 imm) override;
    void xorReg(JitWidth w, RegPtr reg, RegPtr rm) override;
    void xorValue(JitWidth w, RegPtr reg, U32 imm) override;
    void shrReg(JitWidth w, RegPtr reg, RegPtr rm) override;
    void shrValue(JitWidth w, RegPtr reg, U32 imm) override;
    void shrValueWithDest(JitWidth w, RegPtr dst, RegPtr reg, U32 imm) override;
    void shlReg(JitWidth w, RegPtr reg, RegPtr rm) override;
    void shlValue(JitWidth w, RegPtr reg, U32 imm) override;
    void shlValueWithDest(JitWidth w, RegPtr dst, RegPtr reg, U32 imm) override;
    void sarReg(JitWidth w, RegPtr reg, RegPtr rm) override;
    void sarValue(JitWidth w, RegPtr reg, U32 imm) override;
    void sarValueWithDest(JitWidth w, RegPtr dst, RegPtr reg, U32 imm) override;
    void notReg2(JitWidth w, RegPtr reg) override;
    void negReg2(JitWidth w, RegPtr reg) override;
    void bsfReg(JitWidth w, RegPtr reg, RegPtr rm) override;
    void bsrReg(JitWidth w, RegPtr reg, RegPtr rm) override;
    void rolReg(JitWidth w, RegPtr reg, RegPtr rm) override;
    void rolValue(JitWidth w, RegPtr reg, U32 imm) override;
    void rorReg(JitWidth w, RegPtr reg, RegPtr rm) override;
    void rorValue(JitWidth w, RegPtr reg, U32 imm) override;
    void rclReg(JitWidth w, RegPtr reg, RegPtr rm, RegPtr cf) override;
    void rclValue(JitWidth w, RegPtr reg, U32 imm, RegPtr cf) override;
    void rcrReg(JitWidth w, RegPtr reg, RegPtr rm, RegPtr cf) override;
    void rcrValue(JitWidth w, RegPtr reg, U32 imm, RegPtr cf) override;
    void shldReg(JitWidth w, RegPtr reg, RegPtr rm, RegPtr cl) override;
    void shldValue(JitWidth w, RegPtr reg, RegPtr rm, U32 imm) override;
    void shrdReg(JitWidth w, RegPtr reg, RegPtr rm, RegPtr cl) override;
    void shrdValue(JitWidth w, RegPtr reg, RegPtr rm, U32 imm) override;
    void xchgReg(JitWidth w, RegPtr dest, RegPtr src) override;
    void mulReg(JitWidth w, RegPtr reg) override;
    void imulReg(JitWidth w, RegPtr reg) override;
    void imulRRI(JitWidth w, RegPtr dst, RegPtr src, U32 imm, RegPtr overflow = nullptr) override;
    void imulRR(JitWidth w, RegPtr dst, RegPtr src, RegPtr overflow = nullptr) override;
    void divRegRegWithRemainder(JitWidth w, RegPtr dest, RegPtr destHighAndRemainder, RegPtr src) override;
    void idivRegRegWithRemainder(JitWidth w, RegPtr dest, RegPtr destHighAndRemainder, RegPtr src) override;
    void byteSwapReg32(RegPtr reg) override;
    RegPtr compareReg(JitWidth w, RegPtr reg1, RegPtr reg2, JitEvaluate condition, RegPtr result = nullptr) override;
    RegPtr compareValue(JitWidth w, RegPtr reg, U32 value, JitEvaluate condition, RegPtr result = nullptr) override;
    RegPtr testZeroReg(JitWidth w, RegPtr reg, RegPtr result = nullptr) override;
    void absReg(JitWidth w, RegPtr reg) override;
    void clzReg(JitWidth w, RegPtr result, RegPtr reg) override;

    void mov(JitWidth w, RegPtr dest, RegPtr src) override;
    void movzx(JitWidth dstWidth, RegPtr dest, JitWidth srcWidth, RegPtr src) override;
    void movsx(JitWidth dstWidth, RegPtr dest, JitWidth srcWidth, RegPtr src) override;
    void movValue(JitWidth w, RegPtr dst, DYN_PTR_SIZE imm) override;

    // flags (kept in the CPU struct for now)
    RegPtr getReadOnlyFlags(RegPtr tmp = nullptr) override;
    void setFlags(RegPtr flags, U32 mask) override;

    // control flow
    void If(JitWidth w, RegPtr reg) override;
    void IfNot(JitWidth w, RegPtr reg) override;
    void IfTest(JitWidth w, RegPtr reg, RegPtr mask) override;
    void IfTest(JitWidth w, RegPtr reg, U32 mask) override;
    void IfNotTestBit(JitWidth w, RegPtr reg, U32 bitPos) override;
    void IfTestBit(JitWidth w, RegPtr reg, U32 bitPos) override;
    void IfEqual(JitWidth w, RegPtr reg, DYN_PTR_SIZE value) override;
    void IfEqual(JitWidth w, RegPtr reg1, RegPtr reg2) override;
    void IfNotEqual(JitWidth w, RegPtr reg, DYN_PTR_SIZE value) override;
    void IfNotEqual(JitWidth w, RegPtr reg1, RegPtr reg2) override;
    void IfLessThan(JitWidth w, ComparisonType type, RegPtr reg, U32 value) override;
    void IfLessThan(JitWidth w, ComparisonType type, RegPtr reg1, RegPtr reg2) override;
    void IfGreaterThanOrEqual(JitWidth w, ComparisonType type, RegPtr reg1, RegPtr reg2) override;
    void IfGreaterThanOrEqual(JitWidth w, ComparisonType type, RegPtr reg, U32 value) override;
    void IfGreaterThan(JitWidth w, ComparisonType type, RegPtr reg1, RegPtr reg2) override;
    void IfGreaterThan(JitWidth w, ComparisonType type, RegPtr reg, U32 value) override;
    void IfNotCPU(JitWidth w, RegPtr sib, U8 lsl, U32 offset) override;
    void JumpIfCondition(JitConditional condition, U32 address) override;
    U32 MarkJumpLocation() override;
    void Goto(U32 location) override;
    void JumpInBlock(U32 address) override;
    void StartElse() override;
    void EndIf() override;
    void blockExit() override;
    void jmpHost(RegPtr reg) override;
    void jmpHost(DYN_PTR_SIZE address) override;

    // memory
    void readMMU(RegPtr dest, RegPtr index, U32 offset = 0) override;
    void readMMU(RegPtr dest, U32 index) override;
    RegPtr getLinearMemoryBase(RegPtr tmp = nullptr) override;
    void readHost(JitWidth w, MemPtr address, RegPtr result, bool emulatedMemory = true) override;
    void writeHost(JitWidth w, MemPtr address, RegPtr src, bool emulatedMemory = true) override;
    void writeHost(JitWidth w, MemPtr address, U32 imm, bool emulatedMemory = true) override;
    void clearMMUPermissionIfSpansPage(JitWidth w, RegPtr offset, RegPtr reg) override;
    void clearIfSpansPage(JitWidth w, RegPtr offset, RegPtr reg) override;
    RegPtr readCPU(JitWidth w, U32 offset, RegPtr result = nullptr) override;
    RegPtr readCPU(JitWidth w, RegPtr sib, U8 lsl, U32 offset, RegPtr result = nullptr) override;
    void writeCPU(JitWidth w, U32 offset, RegPtr src) override;
    void writeCPU(JitWidth w, RegPtr sib, U8 lsl, U32 offset, RegPtr src) override;
    void writeCPUValue(JitWidth w, U32 offset, DYN_PTR_SIZE value) override;
    void writeCPUValue(JitWidth w, RegPtr sib, U8 lsl, U32 offset, DYN_PTR_SIZE value) override;

    // host calls
    void callHostFunction(void* address, const std::vector<DynParam>& params, bool restoreCache = true, bool saveCache = true) override;
    void callHostFunctionWithResult(RegPtr result, void* address, const std::vector<DynParam>& params) override;

    // Direct flag fusion needs a host flags register. RISC-V has none, so it
    // stays off until compare-and-branch fusion is implemented (phase 3).
    bool supportsDirectCondition(JitConditional condition) override { return false; }
    bool directDoesAffectFlags(DecodedOp* op) override { return false; }
    void direct_cmp(JitWidth, RegPtr, RegPtr) override { kpanic("riscv64 direct_cmp"); }
    void direct_cmp(JitWidth, RegPtr, U32) override { kpanic("riscv64 direct_cmp"); }
    void direct_test(JitWidth, RegPtr, RegPtr) override { kpanic("riscv64 direct_test"); }
    void direct_test(JitWidth, RegPtr, U32) override { kpanic("riscv64 direct_test"); }
    void direct_flags_op(JitWidth, JitFlagOp, RegPtr, RegPtr) override { kpanic("riscv64 direct_flags_op"); }
    void direct_flags_op(JitWidth, JitFlagOp, RegPtr, U32) override { kpanic("riscv64 direct_flags_op"); }
    void direct_flags_op_with_cf(JitWidth, JitCarryOp, RegPtr, RegPtr, RegPtr) override { kpanic("riscv64 direct_flags_op_with_cf"); }
    void direct_flags_op_with_cf(JitWidth, JitCarryOp, RegPtr, U32, RegPtr) override { kpanic("riscv64 direct_flags_op_with_cf"); }
    void direct_neg(JitWidth, RegPtr) override { kpanic("riscv64 direct_neg"); }
    void direct_jump(JitConditional, U32) override { kpanic("riscv64 direct_jump"); }
    void direct_cmov(JitWidth, JitConditional, RegPtr, RegPtr) override { kpanic("riscv64 direct_cmov"); }
    void direct_setcc(JitConditional, RegPtr) override { kpanic("riscv64 direct_setcc"); }

protected:
    U32 markBufferLocation() override;
    U32 getBufferLocation(U32 id) override;
    U32 getBufferSize() override;
    void copyBuffer(U8* dst, U32 size) override;
    U32 getIfJumpSize() override { return (U32)ifLabels.size(); }
    U8* createSyncToHost() override;
    U8* createSyncFromHost() override;
    U8* createBlockExit() override;
    U8* createStartJITCode() override;
    void nakedCall(RegPtr reg) override { as.jalr(ra, R(reg), 0); }
    void nakedReturn() override { as.ret(); }

private:
    static HReg R(const RegPtr& reg) { return (HReg)reg->hardwareReg(); }
    static U32 bits(JitWidth w);

    U8 findTmpReg();
    RegPtr makeTmp(U8 reg);

    // Value of `reg` at `w`, with bit 0 aligned, in a host register. Returns
    // `reg` itself when no shift or extension is needed, otherwise `scratch`.
    enum class Ext { None, Zero, Sign };
    HReg extract(JitWidth w, const RegPtr& reg, HReg scratch, Ext ext);
    // Replace the low `w` bits of dst (bits 8-15 for a high byte register) with
    // the low bits of `value`. `value` must be a scratch register; it is clobbered.
    void insert(JitWidth w, const RegPtr& dst, HReg value);
    void insert(JitWidth w, HReg dst, bool high, HReg value);

    using Op3 = std::function<void(HReg d, HReg a, HReg b)>;
    using OpImm = std::function<bool(HReg d, HReg a, S32 imm)>; // returns false if imm doesn't fit
    void regReg(JitWidth w, const RegPtr& reg, const RegPtr& rm, Op3 op, Ext ext = Ext::None, bool needsZext = true);
    void regRegHost(JitWidth w, const RegPtr& reg, HReg b, Op3 op, Ext ext, bool needsZext);
    void regValue(JitWidth w, const RegPtr& reg, U32 value, Op3 op, OpImm opImm, Ext ext = Ext::None, bool needsZext = true);
    void shiftReg(JitWidth w, const RegPtr& reg, const RegPtr& rm, int kind);
    void shiftValue(JitWidth w, HReg dst, const RegPtr& reg, U32 imm, int kind);
    void rotateNarrow(JitWidth w, const RegPtr& reg, HReg count, bool left);
    void rcl_rcr(JitWidth w, const RegPtr& reg, const RegPtr& rm, U32 imm, bool useImm, const RegPtr& cf, bool left);

    void loadCompareOperands(JitWidth w, const RegPtr& reg1, const RegPtr& reg2, bool isSigned, HReg& a, HReg& b);
    void loadCompareOperands(JitWidth w, const RegPtr& reg, DYN_PTR_SIZE value, bool isSigned, HReg& a, HReg& b);
    void setCompareResult(JitEvaluate condition, HReg result, HReg a, HReg b);
    void ifBranch(Cond skipCond, HReg a, HReg b);
    static bool isSignedEval(JitEvaluate condition);

    // Address of mem in a register (SCR0 unless no arithmetic is needed) plus
    // a 12-bit displacement.
    HReg hostAddress(const MemPtr& mem, S32& disp);
    void cpuAddress(U32 offset, HReg& base, S32& disp);
    void load(JitWidth w, HReg dst, HReg base, S32 disp, bool signExtend = false);
    void store(JitWidth w, HReg src, HReg base, S32 disp);
    void loadCacheFromCPU();
    void writeCacheToCPU();
    void setParams(const std::vector<DynParam>& params);
    std::vector<U8> spillCallerSavedTmps(U8 except);
    void reloadSpilledTmps(const std::vector<U8>& spilled);

    Emitter as;
    std::array<bool, NUMBER_OF_REGS> regUsed{};
    std::vector<Label> ifLabels;
    std::vector<Label> jumpLabels;
    std::vector<Label> bufferLabels;
    BHashTable<U32, Label> opLabels;
    bool resolved = false;
};

// ---------------------------------------------------------------------------
// helpers

U32 JitRiscv64CodeGen::bits(JitWidth w) {
    switch (w) {
    case JitWidth::b8: return 8;
    case JitWidth::b16: return 16;
    case JitWidth::b32: return 32;
    case JitWidth::b64: return 64;
    default: kpanic_fmt("JitRiscv64CodeGen unexpected width %d", (int)w); return 0;
    }
}

HReg JitRiscv64CodeGen::extract(JitWidth w, const RegPtr& reg, HReg scratch, Ext ext) {
    HReg r = R(reg);
    switch (w) {
    case JitWidth::b64:
        return r;
    case JitWidth::b32:
        if (ext == Ext::Sign) {
            as.sext_w(scratch, r);
            return scratch;
        }
        return r; // already zero extended
    case JitWidth::b16:
        if (ext == Ext::Zero) {
            as.zext_h(scratch, r);
            return scratch;
        } else if (ext == Ext::Sign) {
            as.sext_h(scratch, r);
            return scratch;
        }
        return r;
    case JitWidth::b8:
        if (reg->isHigh) {
            as.srli(scratch, r, 8);
            r = scratch;
        }
        if (ext == Ext::Zero) {
            as.andi(scratch, r, 0xff);
            return scratch;
        } else if (ext == Ext::Sign) {
            as.sext_b(scratch, r);
            return scratch;
        }
        return r;
    default:
        kpanic("JitRiscv64CodeGen::extract");
        return r;
    }
}

void JitRiscv64CodeGen::insert(JitWidth w, const RegPtr& dst, HReg value) {
    insert(w, R(dst), w == JitWidth::b8 && dst->isHigh, value);
}

void JitRiscv64CodeGen::insert(JitWidth w, HReg dst, bool high, HReg value) {
    switch (w) {
    case JitWidth::b64:
        as.mv(dst, value);
        break;
    case JitWidth::b32:
        as.zext_w(dst, value);
        break;
    case JitWidth::b16:
        as.zext_h(value, value);
        as.srli(dst, dst, 16);
        as.slli(dst, dst, 16);
        as.or_(dst, dst, value);
        break;
    case JitWidth::b8:
        as.andi(value, value, 0xff);
        if (high) {
            // rotate bits 8-15 down, replace them, then rotate back
            as.rori(dst, dst, 8);
            as.andi(dst, dst, -256);
            as.or_(dst, dst, value);
            as.rori(dst, dst, 56);
        } else {
            as.andi(dst, dst, -256);
            as.or_(dst, dst, value);
        }
        break;
    default:
        kpanic("JitRiscv64CodeGen::insert");
    }
}

void JitRiscv64CodeGen::regRegHost(JitWidth w, const RegPtr& reg, HReg b, Op3 op, Ext ext, bool needsZext) {
    if (w == JitWidth::b64) {
        op(R(reg), R(reg), b);
    } else if (w == JitWidth::b32 && ext != Ext::Sign) {
        op(R(reg), R(reg), b);
        if (needsZext) {
            as.zext_w(R(reg), R(reg));
        }
    } else {
        HReg a = extract(w, reg, SCR0, ext);
        op(SCR0, a, b);
        insert(w, reg, SCR0);
    }
}

void JitRiscv64CodeGen::regReg(JitWidth w, const RegPtr& reg, const RegPtr& rm, Op3 op, Ext ext, bool needsZext) {
    HReg b = R(rm);
    if (w == JitWidth::b8 && rm->isHigh) {
        as.srli(SCR1, b, 8);
        b = SCR1;
    }
    regRegHost(w, reg, b, op, ext, needsZext);
}

void JitRiscv64CodeGen::regValue(JitWidth w, const RegPtr& reg, U32 value, Op3 op, OpImm opImm, Ext ext, bool needsZext) {
    S32 imm = (S32)value;
    if (w == JitWidth::b64 || (w == JitWidth::b32 && ext != Ext::Sign)) {
        // A 64-bit op zero extends the U32 immediate, so a sign-extended
        // 12-bit form only works when the value is small and positive.
        bool immOk = w != JitWidth::b64 || value < 2048;
        if (!opImm || !immOk || !opImm(R(reg), R(reg), imm)) {
            as.li(SCR1, (S64)value);
            op(R(reg), R(reg), SCR1);
        }
        if (w == JitWidth::b32 && needsZext) {
            as.zext_w(R(reg), R(reg));
        }
        return;
    }
    HReg a = extract(w, reg, SCR0, ext);
    if (!opImm || !opImm(SCR0, a, imm)) {
        as.li(SCR1, (S64)value);
        op(SCR0, a, SCR1);
    }
    insert(w, reg, SCR0);
}

// ---------------------------------------------------------------------------
// register allocation

void JitRiscv64CodeGen::preOp(DecodedOp* op) {
    currentOp = op;
    Label label;
    if (!opLabels.get(currentEip, label)) {
        label = as.newLabel();
        opLabels.set(currentEip, label);
    }
    as.bind(label);
}

U8 JitRiscv64CodeGen::findTmpReg() {
    if (disableTmps) {
        kpanic("JitRiscv64CodeGen::findTmpReg disableTmps");
    }
    for (int i = 0; i < NUMBER_OF_TMPS; i++) {
        U8 tmp = tmpRegs[i];
        if (!regUsed[tmp]) {
            regUsed[tmp] = true;
            return tmp;
        }
    }
    kpanic("JitRiscv64CodeGen::getTmpReg ran out of tmp regs");
    return INVALID_REG;
}

RegPtr JitRiscv64CodeGen::makeTmp(U8 reg) {
    return std::shared_ptr<JitReg>(new JitReg(reg, 0xff), [this](JitReg* p) {
        if (p->isLoaded()) {
            regUsed[p->hardwareReg()] = false;
        }
        delete p;
    });
}

RegPtr JitRiscv64CodeGen::getTmpReg() {
    return makeTmp(findTmpReg());
}

RegPtr JitRiscv64CodeGen::getTmpReg8() {
    return getTmpReg();
}

RegPtr JitRiscv64CodeGen::getTmpRegWithHint(S8 hint) {
    return getTmpReg();
}

RegPtr JitRiscv64CodeGen::getTmpRegForCallResult() {
    return getTmpReg();
}

RegPtr JitRiscv64CodeGen::getConditionCalculationReg(U32 index) {
    if (index > 2) {
        kpanic("JitRiscv64CodeGen::getConditionCalculationReg");
    }
    U8 reg = tmpRegs[NUMBER_OF_TMPS - 1 - index];
    if (regUsed[reg]) {
        kpanic_fmt("JitRiscv64CodeGen::getConditionCalculationReg %d in use", index);
    }
    regUsed[reg] = true;
    return makeTmp(reg);
}

RegPtr JitRiscv64CodeGen::getReg(U8 reg, S8 hint, bool load) {
    return std::make_shared<JitReg>(guestRegs[reg], reg);
}

RegPtr JitRiscv64CodeGen::getReg8(U8 reg, bool load) {
    if (reg >= 4) {
        RegPtr result = getReg(reg - 4);
        result->emulatedReg = reg;
        result->isHigh = true;
        return result;
    }
    return getReg(reg);
}

RegPtr JitRiscv64CodeGen::getReadOnlyReg(U8 reg, bool delayed, S8 hint) {
    return std::make_shared<JitReg>(guestRegs[reg], reg);
}

RegPtr JitRiscv64CodeGen::getReadOnlyReg8(U8 reg, bool delayed, S8 hint) {
    U8 hardwareReg = reg >= 4 ? reg - 4 : reg;
    return std::make_shared<JitReg>(guestRegs[hardwareReg], reg, reg >= 4);
}

RegPtr JitRiscv64CodeGen::getTmpReg(U8 reg, bool delayed, S8 hint) {
    if (delayed) {
        auto load = [reg, this]() {
            U8 hardwareReg = findTmpReg();
            as.mv((HReg)hardwareReg, guestRegs[reg]);
            return hardwareReg;
        };
        return std::shared_ptr<JitReg>(new JitReg(0xff, 0xff, load), [this](JitReg* p) {
            if (p->isLoaded()) {
                regUsed[p->hardwareReg()] = false;
            }
            delete p;
        });
    }
    RegPtr result = getTmpReg();
    as.mv(R(result), guestRegs[reg]);
    return result;
}

RegPtr JitRiscv64CodeGen::getTmpReg8(U8 reg, bool delayed, S8 hint) {
    if (reg >= 4) {
        RegPtr result = getTmpReg(reg - 4, delayed, hint);
        result->emulatedReg = reg;
        result->isHigh = true;
        return result;
    }
    return getTmpReg(reg, delayed, hint);
}

RegPtr JitRiscv64CodeGen::getReadOnlySegAddress(U8 reg) {
    return readCPU(JitWidth::b32, CPU::offsetofSegAddress(reg));
}

RegPtr JitRiscv64CodeGen::getTmpSegAddress(U8 reg) {
    return getReadOnlySegAddress(reg);
}

RegPtr JitRiscv64CodeGen::getReadOnlySegValue(U8 reg) {
    return readCPU(JitWidth::b32, CPU::offsetofSegValue(reg));
}

RegPtr JitRiscv64CodeGen::readEip() {
    return readCPU(JitWidth::b32, offsetof(CPU, eip.u32));
}

void JitRiscv64CodeGen::writeEip(RegPtr eip) {
    writeCPU(JitWidth::b32, offsetof(CPU, eip.u32), eip);
}

void JitRiscv64CodeGen::writeEip(U32 eip) {
    writeCPUValue(JitWidth::b32, offsetof(CPU, eip.u32), eip);
}

U32 JitRiscv64CodeGen::getAvailableTmpRegCount() {
    if (disableTmps) {
        return 0;
    }
    U32 result = 0;
    for (int i = 0; i < NUMBER_OF_TMPS; i++) {
        if (!regUsed[tmpRegs[i]]) {
            result++;
        }
    }
    return result;
}

bool JitRiscv64CodeGen::isTmpRegAvailable() {
    return getAvailableTmpRegCount() != 0;
}

void JitRiscv64CodeGen::fixupU32CallResult(RegPtr reg) {
    // The LP64 ABI returns 32-bit values sign extended.
    as.zext_w(R(reg), R(reg));
}

void JitRiscv64CodeGen::emulateSingleOp() {
    writeCurrentEip(0);
    as.jr(regEmulateSingleOp); // does not return here
}

// ---------------------------------------------------------------------------
// arithmetic

static bool fitsImm(S64 v) { return isInt12(v); }

void JitRiscv64CodeGen::addReg(JitWidth w, RegPtr reg, RegPtr rm) {
    regReg(w, reg, rm, [this](HReg d, HReg a, HReg b) { as.add(d, a, b); });
}

void JitRiscv64CodeGen::addRegWithDest(JitWidth w, RegPtr dst, RegPtr reg, RegPtr rm) {
    if (w == JitWidth::b64) {
        as.add(R(dst), R(reg), R(rm));
    } else if (w == JitWidth::b32) {
        as.add(R(dst), R(reg), R(rm));
        as.zext_w(R(dst), R(dst));
    } else {
        JitCodeGen::addRegWithDest(w, dst, reg, rm);
    }
}

void JitRiscv64CodeGen::addValue(JitWidth w, RegPtr reg, U32 imm) {
    regValue(w, reg, imm, [this](HReg d, HReg a, HReg b) { as.add(d, a, b); },
        [this](HReg d, HReg a, S32 imm) { if (!fitsImm(imm)) return false; as.addi(d, a, imm); return true; });
}

void JitRiscv64CodeGen::addValueWithDest(JitWidth w, RegPtr dst, RegPtr reg, U32 imm) {
    if (w == JitWidth::b64 || w == JitWidth::b32) {
        if (w == JitWidth::b64 ? imm < 2048 : fitsImm((S32)imm)) {
            as.addi(R(dst), R(reg), (S32)imm);
        } else {
            as.li(SCR1, (S64)imm);
            as.add(R(dst), R(reg), SCR1);
        }
        if (w == JitWidth::b32) {
            as.zext_w(R(dst), R(dst));
        }
    } else {
        JitCodeGen::addValueWithDest(w, dst, reg, imm);
    }
}

void JitRiscv64CodeGen::subReg(JitWidth w, RegPtr reg, RegPtr rm) {
    regReg(w, reg, rm, [this](HReg d, HReg a, HReg b) { as.sub(d, a, b); });
}

void JitRiscv64CodeGen::subValue(JitWidth w, RegPtr reg, U32 imm) {
    regValue(w, reg, imm, [this](HReg d, HReg a, HReg b) { as.sub(d, a, b); },
        [this](HReg d, HReg a, S32 imm) { if (!fitsImm(-(S64)imm)) return false; as.addi(d, a, -imm); return true; });
}

void JitRiscv64CodeGen::subValueWithDest(JitWidth w, RegPtr dst, RegPtr reg, U32 imm) {
    if (w == JitWidth::b32 || w == JitWidth::b64) {
        if (w == JitWidth::b64 ? imm <= 2048 : fitsImm(-(S64)(S32)imm)) {
            as.addi(R(dst), R(reg), -(S32)imm);
        } else {
            as.li(SCR1, (S64)imm);
            as.sub(R(dst), R(reg), SCR1);
        }
        if (w == JitWidth::b32) {
            as.zext_w(R(dst), R(dst));
        }
    } else {
        JitCodeGen::subValueWithDest(w, dst, reg, imm);
    }
}

void JitRiscv64CodeGen::orReg(JitWidth w, RegPtr reg, RegPtr rm) {
    regReg(w, reg, rm, [this](HReg d, HReg a, HReg b) { as.or_(d, a, b); }, Ext::None, false);
}

void JitRiscv64CodeGen::orValue(JitWidth w, RegPtr reg, U32 imm) {
    regValue(w, reg, imm, [this](HReg d, HReg a, HReg b) { as.or_(d, a, b); },
        [this](HReg d, HReg a, S32 imm) { if (imm < 0 || !fitsImm(imm)) return false; as.ori(d, a, imm); return true; }, Ext::None, false);
}

void JitRiscv64CodeGen::andReg(JitWidth w, RegPtr reg, RegPtr rm) {
    regReg(w, reg, rm, [this](HReg d, HReg a, HReg b) { as.and_(d, a, b); }, Ext::None, false);
}

void JitRiscv64CodeGen::andValue(JitWidth w, RegPtr reg, U32 imm) {
    // andi sign extends its immediate; for a 32-bit value that is harmless
    // because the upper bits of the register are already zero.
    regValue(w, reg, imm, [this](HReg d, HReg a, HReg b) { as.and_(d, a, b); },
        [this](HReg d, HReg a, S32 imm) { if (!fitsImm(imm)) return false; as.andi(d, a, imm); return true; }, Ext::None, false);
}

void JitRiscv64CodeGen::andValue64(RegPtr reg, U64 imm) {
    if (fitsImm((S64)imm)) {
        as.andi(R(reg), R(reg), (S32)(S64)imm);
    } else {
        as.li(SCR1, (S64)imm);
        as.and_(R(reg), R(reg), SCR1);
    }
}

void JitRiscv64CodeGen::andValueWithDest(JitWidth w, RegPtr dst, RegPtr reg, U32 imm) {
    if (w == JitWidth::b32) {
        if (fitsImm((S32)imm)) {
            as.andi(R(dst), R(reg), (S32)imm);
        } else {
            as.li(SCR1, (S64)imm);
            as.and_(R(dst), R(reg), SCR1);
        }
    } else {
        JitCodeGen::andValueWithDest(w, dst, reg, imm);
    }
}

void JitRiscv64CodeGen::xorReg(JitWidth w, RegPtr reg, RegPtr rm) {
    regReg(w, reg, rm, [this](HReg d, HReg a, HReg b) { as.xor_(d, a, b); }, Ext::None, false);
}

void JitRiscv64CodeGen::xorValue(JitWidth w, RegPtr reg, U32 imm) {
    regValue(w, reg, imm, [this](HReg d, HReg a, HReg b) { as.xor_(d, a, b); },
        [this](HReg d, HReg a, S32 imm) { if (imm < 0 || !fitsImm(imm)) return false; as.xori(d, a, imm); return true; }, Ext::None, false);
}

// kind: 0 = shl, 1 = shr, 2 = sar
void JitRiscv64CodeGen::shiftReg(JitWidth w, const RegPtr& reg, const RegPtr& rm, int kind) {
    HReg count = R(rm);
    if (w == JitWidth::b8 && rm->isHigh) {
        as.srli(SCR1, count, 8);
        count = SCR1;
    }
    as.andi(SCR1, count, w == JitWidth::b64 ? 0x3f : 0x1f);
    if (w == JitWidth::b32 && kind == 2) {
        as.sraw(R(reg), R(reg), SCR1);
        as.zext_w(R(reg), R(reg));
        return;
    }
    Ext ext = kind == 1 ? Ext::Zero : (kind == 2 ? Ext::Sign : Ext::None);
    regRegHost(w, reg, SCR1, [this, kind](HReg d, HReg a, HReg b) {
        if (kind == 0) as.sll(d, a, b);
        else if (kind == 1) as.srl(d, a, b);
        else as.sra(d, a, b);
    }, ext, kind == 0);
}

void JitRiscv64CodeGen::shiftValue(JitWidth w, HReg dst, const RegPtr& reg, U32 imm, int kind) {
    imm &= (w == JitWidth::b64) ? 63 : 31;
    if (w == JitWidth::b64) {
        if (kind == 0) as.slli(dst, R(reg), imm);
        else if (kind == 1) as.srli(dst, R(reg), imm);
        else as.srai(dst, R(reg), imm);
    } else if (w == JitWidth::b32) {
        if (kind == 0) {
            as.slli(dst, R(reg), imm);
            as.zext_w(dst, dst);
        } else if (kind == 1) {
            as.srli(dst, R(reg), imm);
        } else {
            as.sraiw(dst, R(reg), imm);
            as.zext_w(dst, dst);
        }
    } else {
        kpanic("JitRiscv64CodeGen::shiftValue");
    }
}

void JitRiscv64CodeGen::shrReg(JitWidth w, RegPtr reg, RegPtr rm) { shiftReg(w, reg, rm, 1); }
void JitRiscv64CodeGen::shlReg(JitWidth w, RegPtr reg, RegPtr rm) { shiftReg(w, reg, rm, 0); }
void JitRiscv64CodeGen::sarReg(JitWidth w, RegPtr reg, RegPtr rm) { shiftReg(w, reg, rm, 2); }

void JitRiscv64CodeGen::shrValue(JitWidth w, RegPtr reg, U32 imm) {
    if (w == JitWidth::b32 || w == JitWidth::b64) {
        shiftValue(w, R(reg), reg, imm, 1);
        return;
    }
    HReg a = extract(w, reg, SCR0, Ext::Zero);
    as.srli(SCR0, a, imm & 31);
    insert(w, reg, SCR0);
}

void JitRiscv64CodeGen::shlValue(JitWidth w, RegPtr reg, U32 imm) {
    if (w == JitWidth::b32 || w == JitWidth::b64) {
        shiftValue(w, R(reg), reg, imm, 0);
        return;
    }
    HReg a = extract(w, reg, SCR0, Ext::None);
    as.slli(SCR0, a, imm & 31);
    insert(w, reg, SCR0);
}

void JitRiscv64CodeGen::sarValue(JitWidth w, RegPtr reg, U32 imm) {
    if (w == JitWidth::b32 || w == JitWidth::b64) {
        shiftValue(w, R(reg), reg, imm, 2);
        return;
    }
    HReg a = extract(w, reg, SCR0, Ext::Sign);
    as.srai(SCR0, a, imm & 31);
    insert(w, reg, SCR0);
}

void JitRiscv64CodeGen::shrValueWithDest(JitWidth w, RegPtr dst, RegPtr reg, U32 imm) {
    if (w == JitWidth::b32 || w == JitWidth::b64) {
        shiftValue(w, R(dst), reg, imm, 1);
    } else {
        JitCodeGen::shrValueWithDest(w, dst, reg, imm);
    }
}

void JitRiscv64CodeGen::shlValueWithDest(JitWidth w, RegPtr dst, RegPtr reg, U32 imm) {
    if (w == JitWidth::b32 || w == JitWidth::b64) {
        shiftValue(w, R(dst), reg, imm, 0);
    } else {
        JitCodeGen::shlValueWithDest(w, dst, reg, imm);
    }
}

void JitRiscv64CodeGen::sarValueWithDest(JitWidth w, RegPtr dst, RegPtr reg, U32 imm) {
    if (w == JitWidth::b32 || w == JitWidth::b64) {
        shiftValue(w, R(dst), reg, imm, 2);
    } else {
        JitCodeGen::sarValueWithDest(w, dst, reg, imm);
    }
}

void JitRiscv64CodeGen::notReg2(JitWidth w, RegPtr reg) {
    if (w == JitWidth::b64) {
        as.not_(R(reg), R(reg));
    } else if (w == JitWidth::b32) {
        as.not_(R(reg), R(reg));
        as.zext_w(R(reg), R(reg));
    } else {
        HReg a = extract(w, reg, SCR0, Ext::None);
        as.not_(SCR0, a);
        insert(w, reg, SCR0);
    }
}

void JitRiscv64CodeGen::negReg2(JitWidth w, RegPtr reg) {
    if (w == JitWidth::b64) {
        as.neg(R(reg), R(reg));
    } else if (w == JitWidth::b32) {
        as.neg(R(reg), R(reg));
        as.zext_w(R(reg), R(reg));
    } else {
        HReg a = extract(w, reg, SCR0, Ext::None);
        as.neg(SCR0, a);
        insert(w, reg, SCR0);
    }
}

void JitRiscv64CodeGen::bsfReg(JitWidth w, RegPtr reg, RegPtr rm) {
    // rm was already checked to be non zero
    if (w == JitWidth::b32) {
        as.ctzw(R(reg), R(rm));
    } else if (w == JitWidth::b16) {
        as.ctzw(SCR0, R(rm)); // a set bit exists in the low 16 bits
        insert(w, reg, SCR0);
    } else {
        kpanic("JitRiscv64CodeGen::bsfReg");
    }
}

void JitRiscv64CodeGen::bsrReg(JitWidth w, RegPtr reg, RegPtr rm) {
    // rm was already checked to be non zero
    if (w == JitWidth::b32) {
        as.clzw(SCR0, R(rm));
        as.xori(R(reg), SCR0, 31);
    } else if (w == JitWidth::b16) {
        as.zext_h(SCR0, R(rm));
        as.clzw(SCR0, SCR0);
        as.xori(SCR0, SCR0, 31);
        insert(w, reg, SCR0);
    } else {
        kpanic("JitRiscv64CodeGen::bsrReg");
    }
}

// r = (x << n) | (x >> (w - n)) for a zero-extended 8/16-bit x
void JitRiscv64CodeGen::rotateNarrow(JitWidth w, const RegPtr& reg, HReg count, bool left) {
    U32 width = bits(w);
    RegPtr tmp = getTmpReg();
    HReg x = extract(w, reg, SCR0, Ext::Zero);
    if (x != SCR0) {
        as.mv(SCR0, x);
    }
    as.andi(SCR1, count, width - 1);
    if (left) {
        as.sll(R(tmp), SCR0, SCR1);
    } else {
        as.srl(R(tmp), SCR0, SCR1);
    }
    as.neg(SCR1, SCR1);
    as.addi(SCR1, SCR1, width);
    if (left) {
        as.srl(SCR0, SCR0, SCR1);
    } else {
        as.sll(SCR0, SCR0, SCR1);
    }
    as.or_(SCR0, SCR0, R(tmp));
    insert(w, reg, SCR0);
}

void JitRiscv64CodeGen::rolReg(JitWidth w, RegPtr reg, RegPtr rm) {
    HReg count = R(rm);
    if (w == JitWidth::b8 && rm->isHigh) {
        kpanic("JitRiscv64CodeGen::rolReg high count");
    }
    if (w == JitWidth::b32) {
        as.andi(SCR1, count, 31);
        as.rolw(R(reg), R(reg), SCR1);
        as.zext_w(R(reg), R(reg));
    } else {
        rotateNarrow(w, reg, count, true);
    }
}

void JitRiscv64CodeGen::rorReg(JitWidth w, RegPtr reg, RegPtr rm) {
    HReg count = R(rm);
    if (w == JitWidth::b8 && rm->isHigh) {
        kpanic("JitRiscv64CodeGen::rorReg high count");
    }
    if (w == JitWidth::b32) {
        as.andi(SCR1, count, 31);
        as.rorw(R(reg), R(reg), SCR1);
        as.zext_w(R(reg), R(reg));
    } else {
        rotateNarrow(w, reg, count, false);
    }
}

void JitRiscv64CodeGen::rolValue(JitWidth w, RegPtr reg, U32 imm) {
    U32 width = bits(w);
    imm %= width;
    if (!imm) {
        return;
    }
    if (w == JitWidth::b32) {
        as.roriw(R(reg), R(reg), 32 - imm);
        as.zext_w(R(reg), R(reg));
        return;
    }
    HReg x = extract(w, reg, SCR0, Ext::Zero);
    as.slli(SCR1, x, imm);
    as.srli(SCR0, x, width - imm);
    as.or_(SCR0, SCR0, SCR1);
    insert(w, reg, SCR0);
}

void JitRiscv64CodeGen::rorValue(JitWidth w, RegPtr reg, U32 imm) {
    U32 width = bits(w);
    imm %= width;
    if (!imm) {
        return;
    }
    if (w == JitWidth::b32) {
        as.roriw(R(reg), R(reg), imm);
        as.zext_w(R(reg), R(reg));
        return;
    }
    HReg x = extract(w, reg, SCR0, Ext::Zero);
    as.srli(SCR1, x, imm);
    as.slli(SCR0, x, width - imm);
    as.or_(SCR0, SCR0, SCR1);
    insert(w, reg, SCR0);
}

// Rotate the (w+1)-bit value CF:x. With v = (cf << w) | x,
//   rcl: (v << n) | (v >> (w + 1 - n))
//   rcr: (v >> n) | (v << (w + 1 - n))
// and the low w bits are the result. 64-bit shifts make n == 0 work too.
void JitRiscv64CodeGen::rcl_rcr(JitWidth w, const RegPtr& reg, const RegPtr& rm, U32 imm, bool useImm, const RegPtr& cf, bool left) {
    U32 width = bits(w);
    RegPtr v = getTmpReg();
    HReg x = extract(w, reg, SCR0, Ext::Zero);
    as.slli(SCR1, R(cf), width);
    as.or_(R(v), x, SCR1);
    if (useImm) {
        U32 n = imm % (width + 1);
        if (!n) {
            return;
        }
        if (left) {
            as.slli(SCR0, R(v), n);
            as.srli(SCR1, R(v), width + 1 - n);
        } else {
            as.srli(SCR0, R(v), n);
            as.slli(SCR1, R(v), width + 1 - n);
        }
    } else {
        RegPtr n = getTmpReg();
        as.andi(R(n), R(rm), 0x1f);
        if (width < 32) {
            as.li(SCR1, width + 1);
            as.remu(R(n), R(n), SCR1);
        }
        if (left) {
            as.sll(SCR0, R(v), R(n));
        } else {
            as.srl(SCR0, R(v), R(n));
        }
        as.neg(SCR1, R(n));
        as.addi(SCR1, SCR1, width + 1);
        if (left) {
            as.srl(SCR1, R(v), SCR1);
        } else {
            as.sll(SCR1, R(v), SCR1);
        }
    }
    as.or_(SCR0, SCR0, SCR1);
    insert(w, reg, SCR0);
}

void JitRiscv64CodeGen::rclReg(JitWidth w, RegPtr reg, RegPtr rm, RegPtr cf) { rcl_rcr(w, reg, rm, 0, false, cf, true); }
void JitRiscv64CodeGen::rclValue(JitWidth w, RegPtr reg, U32 imm, RegPtr cf) { rcl_rcr(w, reg, nullptr, imm, true, cf, true); }
void JitRiscv64CodeGen::rcrReg(JitWidth w, RegPtr reg, RegPtr rm, RegPtr cf) { rcl_rcr(w, reg, rm, 0, false, cf, false); }
void JitRiscv64CodeGen::rcrValue(JitWidth w, RegPtr reg, U32 imm, RegPtr cf) { rcl_rcr(w, reg, nullptr, imm, true, cf, false); }

// shld: 32-bit  result = ((reg << 32 | rm) << n) >> 32
//       16-bit  result = ((reg16 << 32 | rm16 << 16 | rm16) << n) >> 32, which also
//               matches the n >= 16 behavior of the other backends
// shrd: 32-bit  result = (rm << 32 | reg) >> n
//       16-bit  result = (rm16 << 32 | rm16 << 16 | reg16) >> n
void JitRiscv64CodeGen::shldReg(JitWidth w, RegPtr reg, RegPtr rm, RegPtr cl) {
    if (w == JitWidth::b32) {
        as.slli(SCR0, R(reg), 32);
        as.or_(SCR0, SCR0, R(rm));
        as.sll(SCR0, SCR0, R(cl));
        as.srli(R(reg), SCR0, 32);
    } else if (w == JitWidth::b16) {
        as.zext_h(SCR0, R(reg));
        as.slli(SCR0, SCR0, 32);
        as.zext_h(SCR1, R(rm));
        as.or_(SCR0, SCR0, SCR1);
        as.slli(SCR1, SCR1, 16);
        as.or_(SCR0, SCR0, SCR1);
        as.sll(SCR0, SCR0, R(cl));
        as.srli(SCR0, SCR0, 32);
        insert(w, reg, SCR0);
    } else {
        kpanic("JitRiscv64CodeGen::shldReg");
    }
}

void JitRiscv64CodeGen::shldValue(JitWidth w, RegPtr reg, RegPtr rm, U32 imm) {
    imm &= 31;
    if (w == JitWidth::b32) {
        as.slli(SCR0, R(reg), 32);
        as.or_(SCR0, SCR0, R(rm));
        as.slli(SCR0, SCR0, imm);
        as.srli(R(reg), SCR0, 32);
    } else if (w == JitWidth::b16) {
        as.zext_h(SCR0, R(reg));
        as.slli(SCR0, SCR0, 32);
        as.zext_h(SCR1, R(rm));
        as.or_(SCR0, SCR0, SCR1);
        as.slli(SCR1, SCR1, 16);
        as.or_(SCR0, SCR0, SCR1);
        as.slli(SCR0, SCR0, imm);
        as.srli(SCR0, SCR0, 32);
        insert(w, reg, SCR0);
    } else {
        kpanic("JitRiscv64CodeGen::shldValue");
    }
}

void JitRiscv64CodeGen::shrdReg(JitWidth w, RegPtr reg, RegPtr rm, RegPtr cl) {
    if (w == JitWidth::b32) {
        as.slli(SCR0, R(rm), 32);
        as.or_(SCR0, SCR0, R(reg));
        as.srl(SCR0, SCR0, R(cl));
        as.zext_w(R(reg), SCR0);
    } else if (w == JitWidth::b16) {
        as.zext_h(SCR1, R(rm));
        as.slli(SCR0, SCR1, 32);
        as.slli(SCR1, SCR1, 16);
        as.or_(SCR0, SCR0, SCR1);
        as.zext_h(SCR1, R(reg));
        as.or_(SCR0, SCR0, SCR1);
        as.srl(SCR0, SCR0, R(cl));
        insert(w, reg, SCR0);
    } else {
        kpanic("JitRiscv64CodeGen::shrdReg");
    }
}

void JitRiscv64CodeGen::shrdValue(JitWidth w, RegPtr reg, RegPtr rm, U32 imm) {
    imm &= 31;
    if (w == JitWidth::b32) {
        as.slli(SCR0, R(rm), 32);
        as.or_(SCR0, SCR0, R(reg));
        as.srli(SCR0, SCR0, imm);
        as.zext_w(R(reg), SCR0);
    } else if (w == JitWidth::b16) {
        as.zext_h(SCR1, R(rm));
        as.slli(SCR0, SCR1, 32);
        as.slli(SCR1, SCR1, 16);
        as.or_(SCR0, SCR0, SCR1);
        as.zext_h(SCR1, R(reg));
        as.or_(SCR0, SCR0, SCR1);
        as.srli(SCR0, SCR0, imm);
        insert(w, reg, SCR0);
    } else {
        kpanic("JitRiscv64CodeGen::shrdValue");
    }
}

void JitRiscv64CodeGen::xchgReg(JitWidth w, RegPtr dest, RegPtr src) {
    RegPtr tmp = getTmpReg();
    mov(w, tmp, dest);
    mov(w, dest, src);
    mov(w, src, tmp);
}

void JitRiscv64CodeGen::mulReg(JitWidth w, RegPtr reg) {
    HReg eax = guestRegs[0];
    HReg edx = guestRegs[2];
    if (w == JitWidth::b32) {
        // EDX:EAX = (U64)EAX * src
        as.mul(SCR0, eax, R(reg));
        as.srli(edx, SCR0, 32);
        as.zext_w(eax, SCR0);
    } else if (w == JitWidth::b16) {
        // DX:AX = AX * src
        as.zext_h(SCR0, eax);
        as.zext_h(SCR1, R(reg));
        as.mul(SCR0, SCR0, SCR1);
        as.srli(SCR1, SCR0, 16);
        insert(JitWidth::b16, edx, false, SCR1);
        insert(JitWidth::b16, eax, false, SCR0);
    } else if (w == JitWidth::b8) {
        // AX = AL * src
        HReg src = extract(JitWidth::b8, reg, SCR1, Ext::Zero);
        as.andi(SCR0, eax, 0xff);
        as.mul(SCR0, SCR0, src);
        insert(JitWidth::b16, eax, false, SCR0);
    } else {
        kpanic("JitRiscv64CodeGen::mulReg");
    }
}

void JitRiscv64CodeGen::imulReg(JitWidth w, RegPtr reg) {
    HReg eax = guestRegs[0];
    HReg edx = guestRegs[2];
    if (w == JitWidth::b32) {
        // EDX:EAX = (S64)EAX * src
        as.sext_w(SCR0, eax);
        as.sext_w(SCR1, R(reg));
        as.mul(SCR0, SCR0, SCR1);
        as.srli(edx, SCR0, 32);
        as.zext_w(eax, SCR0);
    } else if (w == JitWidth::b16) {
        as.sext_h(SCR0, eax);
        as.sext_h(SCR1, R(reg));
        as.mul(SCR0, SCR0, SCR1);
        as.srli(SCR1, SCR0, 16);
        insert(JitWidth::b16, edx, false, SCR1);
        insert(JitWidth::b16, eax, false, SCR0);
    } else if (w == JitWidth::b8) {
        HReg src = extract(JitWidth::b8, reg, SCR1, Ext::Sign);
        as.sext_b(SCR0, eax);
        as.mul(SCR0, SCR0, src);
        insert(JitWidth::b16, eax, false, SCR0);
    } else {
        kpanic("JitRiscv64CodeGen::imulReg");
    }
}

void JitRiscv64CodeGen::imulRR(JitWidth w, RegPtr dst, RegPtr src, RegPtr overflow) {
    if (w == JitWidth::b32) {
        as.sext_w(SCR0, R(dst));
        as.sext_w(SCR1, R(src));
        as.mul(SCR0, SCR0, SCR1);
        if (overflow) {
            as.srli(R(overflow), SCR0, 32);
        }
        as.zext_w(R(dst), SCR0);
    } else if (w == JitWidth::b16) {
        if (overflow) {
            kpanic("JitRiscv64CodeGen::imulRR overflow");
        }
        as.sext_h(SCR0, R(dst));
        as.sext_h(SCR1, R(src));
        as.mul(SCR0, SCR0, SCR1);
        insert(w, dst, SCR0);
    } else {
        kpanic("JitRiscv64CodeGen::imulRR");
    }
}

void JitRiscv64CodeGen::imulRRI(JitWidth w, RegPtr dst, RegPtr src, U32 imm, RegPtr overflow) {
    if (w == JitWidth::b32) {
        as.sext_w(SCR0, R(src));
        as.li(SCR1, (S32)imm);
        as.mul(SCR0, SCR0, SCR1);
        if (overflow) {
            as.srli(R(overflow), SCR0, 32);
        }
        as.zext_w(R(dst), SCR0);
    } else if (w == JitWidth::b16) {
        as.sext_h(SCR0, R(src));
        as.li(SCR1, (S16)imm);
        as.mul(SCR0, SCR0, SCR1);
        if (overflow) {
            // bits 16-31 of the product
            as.srli(R(overflow), SCR0, 16);
            as.zext_h(R(overflow), R(overflow));
        }
        insert(w, dst, SCR0);
    } else {
        kpanic("JitRiscv64CodeGen::imulRRI");
    }
}

void JitRiscv64CodeGen::divRegRegWithRemainder(JitWidth w, RegPtr dest, RegPtr destHigh, RegPtr src) {
    // the caller already checked for a zero divisor and quotient overflow
    if (w == JitWidth::b32) {
        // U64 num = ((U64)EDX << 32) | EAX
        RegPtr q = getTmpReg();
        as.slli(SCR0, R(destHigh), 32);
        as.or_(SCR0, SCR0, R(dest));
        as.divu(R(q), SCR0, R(src));
        as.remu(SCR0, SCR0, R(src));
        as.zext_w(R(dest), R(q));
        as.zext_w(R(destHigh), SCR0);
    } else if (w == JitWidth::b16) {
        // U32 num = ((U32)DX << 16) | AX
        RegPtr q = getTmpReg();
        as.zext_h(SCR1, R(destHigh));
        as.slli(SCR1, SCR1, 16);
        as.zext_h(SCR0, R(dest));
        as.or_(SCR0, SCR0, SCR1);
        as.zext_h(SCR1, R(src));
        as.divu(R(q), SCR0, SCR1);
        as.remu(SCR0, SCR0, SCR1);
        insert(w, destHigh, SCR0);
        as.mv(SCR0, R(q));
        insert(w, dest, SCR0);
    } else if (w == JitWidth::b8) {
        // AL = AX / src, AH = AX % src
        RegPtr q = getTmpReg();
        HReg divisor = extract(JitWidth::b8, src, SCR1, Ext::Zero);
        as.zext_h(SCR0, R(dest));
        as.divu(R(q), SCR0, divisor);
        as.remu(SCR0, SCR0, divisor);
        insert(JitWidth::b8, R(dest), true, SCR0);
        as.mv(SCR0, R(q));
        insert(JitWidth::b8, R(dest), false, SCR0);
    } else {
        kpanic("JitRiscv64CodeGen::divRegRegWithRemainder");
    }
}

void JitRiscv64CodeGen::idivRegRegWithRemainder(JitWidth w, RegPtr dest, RegPtr destHigh, RegPtr src) {
    if (w == JitWidth::b32) {
        // S64 num = ((S64)EDX << 32) | EAX
        RegPtr q = getTmpReg();
        as.slli(SCR0, R(destHigh), 32);
        as.or_(SCR0, SCR0, R(dest));
        as.sext_w(SCR1, R(src));
        as.div(R(q), SCR0, SCR1);
        as.rem(SCR0, SCR0, SCR1);
        as.zext_w(R(dest), R(q));
        as.zext_w(R(destHigh), SCR0);
    } else if (w == JitWidth::b16) {
        // S32 num = ((S32)DX << 16) | AX
        RegPtr q = getTmpReg();
        as.slli(SCR1, R(destHigh), 16);
        as.zext_h(SCR0, R(dest));
        as.or_(SCR0, SCR0, SCR1);
        as.sext_w(SCR0, SCR0);
        as.sext_h(SCR1, R(src));
        as.div(R(q), SCR0, SCR1);
        as.rem(SCR0, SCR0, SCR1);
        insert(w, destHigh, SCR0);
        as.mv(SCR0, R(q));
        insert(w, dest, SCR0);
    } else if (w == JitWidth::b8) {
        RegPtr q = getTmpReg();
        HReg divisor = extract(JitWidth::b8, src, SCR1, Ext::Sign);
        as.sext_h(SCR0, R(dest));
        as.div(R(q), SCR0, divisor);
        as.rem(SCR0, SCR0, divisor);
        insert(JitWidth::b8, R(dest), true, SCR0);
        as.mv(SCR0, R(q));
        insert(JitWidth::b8, R(dest), false, SCR0);
    } else {
        kpanic("JitRiscv64CodeGen::idivRegRegWithRemainder");
    }
}

void JitRiscv64CodeGen::byteSwapReg32(RegPtr reg) {
    as.rev8(R(reg), R(reg));
    as.srli(R(reg), R(reg), 32);
}

void JitRiscv64CodeGen::absReg(JitWidth w, RegPtr reg) {
    HReg r = R(reg);
    if (w == JitWidth::b64) {
        as.srai(SCR1, r, 63);
        as.xor_(r, r, SCR1);
        as.sub(r, r, SCR1);
        return;
    }
    // Like the other backends, narrow widths are sign extended to 32 bits first.
    if (w == JitWidth::b16) {
        as.sext_h(SCR0, r);
    } else if (w == JitWidth::b8) {
        as.sext_b(SCR0, extract(JitWidth::b8, reg, SCR0, Ext::None));
    } else if (w == JitWidth::b32) {
        as.sext_w(SCR0, r);
    } else {
        kpanic("JitRiscv64CodeGen::absReg");
    }
    as.srai(SCR1, SCR0, 63);
    as.xor_(SCR0, SCR0, SCR1);
    as.sub(SCR0, SCR0, SCR1);
    as.zext_w(r, SCR0);
}

void JitRiscv64CodeGen::clzReg(JitWidth w, RegPtr result, RegPtr reg) {
    if (w == JitWidth::b32) {
        as.clzw(R(result), R(reg));
    } else if (w == JitWidth::b64) {
        as.clz(R(result), R(reg));
    } else {
        kpanic("JitRiscv64CodeGen::clzReg");
    }
}

// ---------------------------------------------------------------------------
// compares

bool JitRiscv64CodeGen::isSignedEval(JitEvaluate condition) {
    switch (condition) {
    case JitEvaluate::LESS_THAN_SIGNED:
    case JitEvaluate::LESS_THAN_EQUAL_SIGNED:
    case JitEvaluate::GREATER_THAN_SIGNED:
    case JitEvaluate::GREATER_THAN_EQUAL_SIGNED:
        return true;
    default:
        return false;
    }
}

void JitRiscv64CodeGen::loadCompareOperands(JitWidth w, const RegPtr& reg1, const RegPtr& reg2, bool isSigned, HReg& a, HReg& b) {
    Ext ext = isSigned ? Ext::Sign : Ext::Zero;
    a = extract(w, reg1, SCR0, ext);
    b = extract(w, reg2, SCR1, ext);
}

void JitRiscv64CodeGen::loadCompareOperands(JitWidth w, const RegPtr& reg, DYN_PTR_SIZE value, bool isSigned, HReg& a, HReg& b) {
    Ext ext = isSigned ? Ext::Sign : Ext::Zero;
    a = extract(w, reg, SCR0, ext);
    S64 v;
    switch (w) {
    case JitWidth::b8: v = isSigned ? (S64)(S8)value : (S64)(U8)value; break;
    case JitWidth::b16: v = isSigned ? (S64)(S16)value : (S64)(U16)value; break;
    case JitWidth::b32: v = isSigned ? (S64)(S32)value : (S64)(U32)value; break;
    default: v = (S64)value; break;
    }
    if (v == 0) {
        b = zero;
    } else {
        as.li(SCR1, v);
        b = SCR1;
    }
}

void JitRiscv64CodeGen::setCompareResult(JitEvaluate condition, HReg result, HReg a, HReg b) {
    switch (condition) {
    case JitEvaluate::EQUALS:
        as.xor_(SCR0, a, b);
        as.seqz(result, SCR0);
        break;
    case JitEvaluate::NOT_EQUALS:
        as.xor_(SCR0, a, b);
        as.snez(result, SCR0);
        break;
    case JitEvaluate::LESS_THAN_UNSIGNED:
        as.sltu(result, a, b);
        break;
    case JitEvaluate::GREATER_THAN_EQUAL_UNSIGNED:
        as.sltu(result, a, b);
        as.xori(result, result, 1);
        break;
    case JitEvaluate::GREATER_THAN_UNSIGNED:
        as.sltu(result, b, a);
        break;
    case JitEvaluate::LESS_THAN_EQUAL_UNSIGNED:
        as.sltu(result, b, a);
        as.xori(result, result, 1);
        break;
    case JitEvaluate::LESS_THAN_SIGNED:
        as.slt(result, a, b);
        break;
    case JitEvaluate::GREATER_THAN_EQUAL_SIGNED:
        as.slt(result, a, b);
        as.xori(result, result, 1);
        break;
    case JitEvaluate::GREATER_THAN_SIGNED:
        as.slt(result, b, a);
        break;
    case JitEvaluate::LESS_THAN_EQUAL_SIGNED:
        as.slt(result, b, a);
        as.xori(result, result, 1);
        break;
    default:
        kpanic_fmt("JitRiscv64CodeGen::compare unknown condition %d", (int)condition);
    }
}

RegPtr JitRiscv64CodeGen::compareReg(JitWidth w, RegPtr reg1, RegPtr reg2, JitEvaluate condition, RegPtr result) {
    if (!result) {
        result = getTmpReg();
    }
    HReg a, b;
    loadCompareOperands(w, reg1, reg2, isSignedEval(condition), a, b);
    setCompareResult(condition, R(result), a, b);
    return result;
}

RegPtr JitRiscv64CodeGen::compareValue(JitWidth w, RegPtr reg, U32 value, JitEvaluate condition, RegPtr result) {
    if (!result) {
        result = getTmpReg();
    }
    HReg a, b;
    loadCompareOperands(w, reg, value, isSignedEval(condition), a, b);
    setCompareResult(condition, R(result), a, b);
    return result;
}

RegPtr JitRiscv64CodeGen::testZeroReg(JitWidth w, RegPtr reg, RegPtr result) {
    if (!result) {
        result = getTmpReg();
    }
    HReg a = extract(w, reg, SCR0, Ext::Zero);
    as.seqz(R(result), a);
    return result;
}

// ---------------------------------------------------------------------------
// moves

void JitRiscv64CodeGen::mov(JitWidth w, RegPtr dest, RegPtr src) {
    if (w == JitWidth::b32 || w == JitWidth::b64) {
        as.mv(R(dest), R(src));
    } else {
        HReg value = extract(w, src, SCR0, Ext::None);
        if (value != SCR0) {
            as.mv(SCR0, value);
        }
        insert(w, dest, SCR0);
    }
}

void JitRiscv64CodeGen::movValue(JitWidth w, RegPtr dst, DYN_PTR_SIZE imm) {
    if (w == JitWidth::b64) {
        as.li(R(dst), (S64)imm);
    } else if (w == JitWidth::b32) {
        as.li(R(dst), (S64)(U32)imm);
    } else {
        as.li(SCR0, (S64)(imm & (w == JitWidth::b16 ? 0xffff : 0xff)));
        insert(w, dst, SCR0);
    }
}

void JitRiscv64CodeGen::movzx(JitWidth dstWidth, RegPtr dest, JitWidth srcWidth, RegPtr src) {
    if (dstWidth == JitWidth::b32) {
        if (srcWidth == JitWidth::b32) {
            as.mv(R(dest), R(src));
        } else if (srcWidth == JitWidth::b16) {
            as.zext_h(R(dest), R(src));
        } else if (srcWidth == JitWidth::b8) {
            if (src->isHigh) {
                as.srli(R(dest), R(src), 8);
                as.andi(R(dest), R(dest), 0xff);
            } else {
                as.andi(R(dest), R(src), 0xff);
            }
            dest->isHigh = false; // for when src == dest
        } else {
            kpanic_fmt("JitRiscv64CodeGen::movzx %d <= %d", (int)dstWidth, (int)srcWidth);
        }
    } else if (dstWidth == JitWidth::b16 && srcWidth == JitWidth::b8) {
        HReg value = extract(JitWidth::b8, src, SCR0, Ext::Zero);
        insert(JitWidth::b16, R(dest), false, value);
        dest->isHigh = false;
    } else {
        kpanic_fmt("JitRiscv64CodeGen::movzx %d <= %d", (int)dstWidth, (int)srcWidth);
    }
}

void JitRiscv64CodeGen::movsx(JitWidth dstWidth, RegPtr dest, JitWidth srcWidth, RegPtr src) {
    if (dstWidth == JitWidth::b32) {
        if (srcWidth == JitWidth::b16) {
            as.sext_h(R(dest), R(src));
        } else if (srcWidth == JitWidth::b8) {
            as.sext_b(R(dest), extract(JitWidth::b8, src, SCR0, Ext::None));
            dest->isHigh = false;
        } else {
            kpanic_fmt("JitRiscv64CodeGen::movsx %d <= %d", (int)dstWidth, (int)srcWidth);
        }
        as.zext_w(R(dest), R(dest));
    } else if (dstWidth == JitWidth::b16 && srcWidth == JitWidth::b8) {
        HReg value = extract(JitWidth::b8, src, SCR0, Ext::Sign);
        insert(JitWidth::b16, R(dest), false, value);
        dest->isHigh = false;
    } else {
        kpanic_fmt("JitRiscv64CodeGen::movsx %d <= %d", (int)dstWidth, (int)srcWidth);
    }
}

// ---------------------------------------------------------------------------
// flags

RegPtr JitRiscv64CodeGen::getReadOnlyFlags(RegPtr tmp) {
    return readCPU(JitWidth::b32, offsetof(CPU, flags), tmp);
}

void JitRiscv64CodeGen::setFlags(RegPtr flags, U32 mask) {
    RegPtr current = readCPU(JitWidth::b32, offsetof(CPU, flags));
    as.li(SCR1, (S64)(U32)~mask);
    as.and_(R(current), R(current), SCR1);
    as.li(SCR1, (S64)mask);
    as.and_(SCR0, R(flags), SCR1);
    as.or_(R(current), R(current), SCR0);
    writeCPU(JitWidth::b32, offsetof(CPU, flags), current);
    currentLazyFlags = FLAGS_NONE;
    storeLazyFlagType(FLAGS_NONE);
}

// ---------------------------------------------------------------------------
// control flow
//
// Structured If blocks branch to an EndIf/StartElse label whose distance is
// unknown, so they use the inverted-branch + jal form (+-1 MiB).

void JitRiscv64CodeGen::ifBranch(Cond skipCond, HReg a, HReg b) {
    Label label = as.newLabel();
    ifLabels.push_back(label);
    as.branchFar(skipCond, a, b, label);
}

void JitRiscv64CodeGen::If(JitWidth w, RegPtr reg) {
    ifBranch(Cond::EQ, extract(w, reg, SCR0, Ext::Zero), zero);
}

void JitRiscv64CodeGen::IfNot(JitWidth w, RegPtr reg) {
    ifBranch(Cond::NE, extract(w, reg, SCR0, Ext::Zero), zero);
}

void JitRiscv64CodeGen::IfTest(JitWidth w, RegPtr reg, RegPtr mask) {
    HReg a = extract(w, reg, SCR0, Ext::None);
    HReg b = extract(w, mask, SCR1, Ext::None);
    as.and_(SCR0, a, b);
    if (w == JitWidth::b8) {
        as.andi(SCR0, SCR0, 0xff);
    } else if (w == JitWidth::b16) {
        as.zext_h(SCR0, SCR0);
    }
    ifBranch(Cond::EQ, SCR0, zero);
}

void JitRiscv64CodeGen::IfTest(JitWidth w, RegPtr reg, U32 value) {
    if (w == JitWidth::b8 && reg->isHigh) {
        value <<= 8;
    }
    if (fitsImm((S32)value)) {
        as.andi(SCR0, R(reg), (S32)value);
    } else {
        as.li(SCR1, (S64)value);
        as.and_(SCR0, R(reg), SCR1);
    }
    ifBranch(Cond::EQ, SCR0, zero);
}

void JitRiscv64CodeGen::IfNotTestBit(JitWidth w, RegPtr reg, U32 bitPos) {
    if (bitPos > 63 || (w == JitWidth::b8 && reg->isHigh)) {
        kpanic("JitRiscv64CodeGen::IfNotTestBit");
    }
    as.srli(SCR0, R(reg), bitPos);
    as.andi(SCR0, SCR0, 1);
    ifBranch(Cond::NE, SCR0, zero);
}

void JitRiscv64CodeGen::IfTestBit(JitWidth w, RegPtr reg, U32 bitPos) {
    if (bitPos > 63 || (w == JitWidth::b8 && reg->isHigh)) {
        kpanic("JitRiscv64CodeGen::IfTestBit");
    }
    as.srli(SCR0, R(reg), bitPos);
    as.andi(SCR0, SCR0, 1);
    ifBranch(Cond::EQ, SCR0, zero);
}

void JitRiscv64CodeGen::IfEqual(JitWidth w, RegPtr reg, DYN_PTR_SIZE value) {
    HReg a, b;
    loadCompareOperands(w, reg, value, false, a, b);
    ifBranch(Cond::NE, a, b);
}

void JitRiscv64CodeGen::IfEqual(JitWidth w, RegPtr reg1, RegPtr reg2) {
    HReg a, b;
    loadCompareOperands(w, reg1, reg2, false, a, b);
    ifBranch(Cond::NE, a, b);
}

void JitRiscv64CodeGen::IfNotEqual(JitWidth w, RegPtr reg, DYN_PTR_SIZE value) {
    HReg a, b;
    loadCompareOperands(w, reg, value, false, a, b);
    ifBranch(Cond::EQ, a, b);
}

void JitRiscv64CodeGen::IfNotEqual(JitWidth w, RegPtr reg1, RegPtr reg2) {
    HReg a, b;
    loadCompareOperands(w, reg1, reg2, false, a, b);
    ifBranch(Cond::EQ, a, b);
}

// The body runs when the condition holds, so each branch skips on the inverse.
void JitRiscv64CodeGen::IfLessThan(JitWidth w, ComparisonType type, RegPtr reg, U32 value) {
    HReg a, b;
    bool isSigned = type == ComparisonType::Signed;
    loadCompareOperands(w, reg, value, isSigned, a, b);
    ifBranch(isSigned ? Cond::GE : Cond::GEU, a, b);
}

void JitRiscv64CodeGen::IfLessThan(JitWidth w, ComparisonType type, RegPtr reg1, RegPtr reg2) {
    HReg a, b;
    bool isSigned = type == ComparisonType::Signed;
    loadCompareOperands(w, reg1, reg2, isSigned, a, b);
    ifBranch(isSigned ? Cond::GE : Cond::GEU, a, b);
}

void JitRiscv64CodeGen::IfGreaterThanOrEqual(JitWidth w, ComparisonType type, RegPtr reg1, RegPtr reg2) {
    HReg a, b;
    bool isSigned = type == ComparisonType::Signed;
    loadCompareOperands(w, reg1, reg2, isSigned, a, b);
    ifBranch(isSigned ? Cond::LT : Cond::LTU, a, b);
}

void JitRiscv64CodeGen::IfGreaterThanOrEqual(JitWidth w, ComparisonType type, RegPtr reg, U32 value) {
    HReg a, b;
    bool isSigned = type == ComparisonType::Signed;
    loadCompareOperands(w, reg, value, isSigned, a, b);
    ifBranch(isSigned ? Cond::LT : Cond::LTU, a, b);
}

// a > b  <=>  b < a; skip when b >= a
void JitRiscv64CodeGen::IfGreaterThan(JitWidth w, ComparisonType type, RegPtr reg1, RegPtr reg2) {
    HReg a, b;
    bool isSigned = type == ComparisonType::Signed;
    loadCompareOperands(w, reg1, reg2, isSigned, a, b);
    ifBranch(isSigned ? Cond::GE : Cond::GEU, b, a);
}

void JitRiscv64CodeGen::IfGreaterThan(JitWidth w, ComparisonType type, RegPtr reg, U32 value) {
    HReg a, b;
    bool isSigned = type == ComparisonType::Signed;
    loadCompareOperands(w, reg, value, isSigned, a, b);
    ifBranch(isSigned ? Cond::GE : Cond::GEU, b, a);
}

void JitRiscv64CodeGen::IfNotCPU(JitWidth w, RegPtr sib, U8 lsl, U32 offset) {
    RegPtr tmp = readCPU(w, sib, lsl, offset);
    IfNot(w, tmp);
}

void JitRiscv64CodeGen::StartElse() {
    Label label = as.newLabel();
    as.j(label);
    as.bind(ifLabels.back());
    ifLabels.pop_back();
    ifLabels.push_back(label);
}

void JitRiscv64CodeGen::EndIf() {
    as.bind(ifLabels.back());
    ifLabels.pop_back();
}

void JitRiscv64CodeGen::JumpIfCondition(JitConditional condition, U32 address) {
    bool negative = false;
    RegPtr reg;
    preIfCondition(condition, negative, reg);
    Label label;
    if (!opLabels.get(address, label)) {
        label = as.newLabel();
        opLabels.set(address, label);
    }
    as.branchFar(negative ? Cond::EQ : Cond::NE, R(reg), zero, label);
}

void JitRiscv64CodeGen::JumpInBlock(U32 address) {
    Label label;
    if (!opLabels.get(address, label)) {
        label = as.newLabel();
        opLabels.set(address, label);
    }
    as.j(label);
}

U32 JitRiscv64CodeGen::MarkJumpLocation() {
    Label label = as.newLabel();
    as.bind(label);
    jumpLabels.push_back(label);
    return (U32)jumpLabels.size();
}

void JitRiscv64CodeGen::Goto(U32 location) {
    as.j(jumpLabels[location - 1]);
}

void JitRiscv64CodeGen::jmpHost(RegPtr reg) {
    as.jr(R(reg));
}

void JitRiscv64CodeGen::jmpHost(DYN_PTR_SIZE address) {
    as.li(SCR0, (S64)address);
    as.jr(SCR0);
}

void JitRiscv64CodeGen::blockExit() {
    as.li(SCR0, (S64)(DYN_PTR_SIZE)cpu->thread->process->blockExit);
    as.jr(SCR0);
}

// ---------------------------------------------------------------------------
// memory

void JitRiscv64CodeGen::load(JitWidth w, HReg dst, HReg base, S32 disp, bool signExtend) {
    switch (w) {
    case JitWidth::b8: if (signExtend) as.lb(dst, base, disp); else as.lbu(dst, base, disp); break;
    case JitWidth::b16: if (signExtend) as.lh(dst, base, disp); else as.lhu(dst, base, disp); break;
    case JitWidth::b32: if (signExtend) as.lw(dst, base, disp); else as.lwu(dst, base, disp); break;
    case JitWidth::b64: as.ld(dst, base, disp); break;
    default: kpanic_fmt("JitRiscv64CodeGen::load width %d", (int)w);
    }
}

void JitRiscv64CodeGen::store(JitWidth w, HReg src, HReg base, S32 disp) {
    switch (w) {
    case JitWidth::b8: as.sb(src, base, disp); break;
    case JitWidth::b16: as.sh(src, base, disp); break;
    case JitWidth::b32: as.sw(src, base, disp); break;
    case JitWidth::b64: as.sd(src, base, disp); break;
    default: kpanic_fmt("JitRiscv64CodeGen::store width %d", (int)w);
    }
}

HReg JitRiscv64CodeGen::hostAddress(const MemPtr& mem, S32& disp) {
    // Uses only SCR0 so the caller can keep a value in SCR1.
    U32 offset = mem->offset;
    bool smallOffset = offset < 2048;
    if (!mem->sib && smallOffset) {
        disp = (S32)offset;
        return mem->rm ? R(mem->rm) : zero;
    }
    if (mem->sib && !mem->rm && !mem->lsl && smallOffset) {
        disp = (S32)offset;
        return R(mem->sib);
    }
    disp = 0;
    HReg base = zero;
    if (!smallOffset) {
        as.li(SCR0, (S64)(U64)offset);
        base = SCR0;
    } else {
        disp = (S32)offset;
    }
    if (mem->rm) {
        if (base == zero) {
            as.mv(SCR0, R(mem->rm));
        } else {
            as.add(SCR0, SCR0, R(mem->rm));
        }
        base = SCR0;
    }
    if (mem->sib) {
        HReg acc = base;
        switch (mem->lsl) {
        case 0: as.add(SCR0, acc, R(mem->sib)); break;
        case 1: as.sh1add(SCR0, R(mem->sib), acc); break;
        case 2: as.sh2add(SCR0, R(mem->sib), acc); break;
        case 3: as.sh3add(SCR0, R(mem->sib), acc); break;
        default: kpanic("JitRiscv64CodeGen::hostAddress lsl");
        }
        base = SCR0;
    }
    return base;
}

void JitRiscv64CodeGen::readHost(JitWidth w, MemPtr mem, RegPtr dest, bool emulatedMemory) {
    S32 disp;
    HReg base = hostAddress(mem, disp);
    if ((w == JitWidth::b8 || w == JitWidth::b16) && isGuestHostReg(dest->hardwareReg())) {
        // a narrow load into a guest register keeps the bits above it
        load(w, SCR1, base, disp);
        insert(w, dest, SCR1);
        return;
    }
    if (w == JitWidth::b8 && dest->isHigh) {
        kpanic("JitRiscv64CodeGen::readHost unexpected dest");
    }
    load(w, R(dest), base, disp);
}

void JitRiscv64CodeGen::writeHost(JitWidth w, MemPtr mem, RegPtr src, bool emulatedMemory) {
    HReg value = R(src);
    if (w == JitWidth::b8 && src->isHigh) {
        as.srli(SCR1, value, 8);
        value = SCR1;
    }
    S32 disp;
    HReg base = hostAddress(mem, disp);
    store(w, value, base, disp);
}

void JitRiscv64CodeGen::writeHost(JitWidth w, MemPtr mem, U32 imm, bool emulatedMemory) {
    HReg value = zero;
    if (imm) {
        as.li(SCR1, (S64)imm);
        value = SCR1;
    }
    S32 disp;
    HReg base = hostAddress(mem, disp);
    store(w, value, base, disp);
}

void JitRiscv64CodeGen::readMMU(RegPtr dest, RegPtr index, U32 offset) {
    as.sh3add(SCR0, R(index), regMMU);
    if (offset < 2048) {
        as.ld(R(dest), SCR0, (S32)offset);
    } else {
        as.li(SCR1, (S64)offset);
        as.add(SCR0, SCR0, SCR1);
        as.ld(R(dest), SCR0, 0);
    }
}

void JitRiscv64CodeGen::readMMU(RegPtr dest, U32 index) {
    U64 offset = (U64)index * 8;
    if (offset < 2048) {
        as.ld(R(dest), regMMU, (S32)offset);
    } else {
        as.li(SCR0, (S64)offset);
        as.add(SCR0, SCR0, regMMU);
        as.ld(R(dest), SCR0, 0);
    }
}

RegPtr JitRiscv64CodeGen::getLinearMemoryBase(RegPtr tmp) {
    if (!tmp) {
        tmp = getTmpReg();
    }
    constexpr S64 offset = (S64)offsetof(KMemoryData, linearMemoryBase) - (S64)offsetof(KMemoryData, mmu);
    if (isInt12(offset)) {
        as.ld(R(tmp), regMMU, (S32)offset);
    } else {
        as.li(SCR0, offset);
        as.add(SCR0, SCR0, regMMU);
        as.ld(R(tmp), SCR0, 0);
    }
    return tmp;
}

void JitRiscv64CodeGen::clearMMUPermissionIfSpansPage(JitWidth w, RegPtr offset, RegPtr reg) {
    U32 limit;
    switch (w) {
    case JitWidth::b16: limit = 0xFFF; break;
    case JitWidth::b32: limit = 0xFFD; break;
    case JitWidth::b64: limit = 0xFF9; break;
    case JitWidth::b128: limit = 0xFF1; break;
    case JitWidth::b256: limit = 0xFE1; break;
    default: kpanic_fmt("JitRiscv64CodeGen::clearMMUPermissionIfSpansPage width %d", (int)w); return;
    }
    // reg = (offset < limit) ? reg : 0, without a branch
    as.li(SCR1, limit);
    as.sltu(SCR0, R(offset), SCR1);
    as.neg(SCR0, SCR0);
    as.and_(R(reg), R(reg), SCR0);
}

void JitRiscv64CodeGen::clearIfSpansPage(JitWidth w, RegPtr offset, RegPtr reg) {
    clearMMUPermissionIfSpansPage(w, offset, reg);
}

void JitRiscv64CodeGen::cpuAddress(U32 offset, HReg& base, S32& disp) {
    if (offset < 2048) {
        base = regCPU;
        disp = (S32)offset;
    } else {
        as.li(SCR0, (S64)offset);
        as.add(SCR0, SCR0, regCPU);
        base = SCR0;
        disp = 0;
    }
}

RegPtr JitRiscv64CodeGen::readCPU(JitWidth w, U32 offset, RegPtr result) {
    if (!result) {
        result = getTmpReg();
    }
    HReg base;
    S32 disp;
    cpuAddress(offset, base, disp);
    load(w, R(result), base, disp);
    return result;
}

RegPtr JitRiscv64CodeGen::readCPU(JitWidth w, RegPtr sib, U8 lsl, U32 offset, RegPtr result) {
    if (!result) {
        result = getTmpReg();
    }
    S32 disp;
    HReg base = hostAddress(createMemPtr(std::make_shared<JitReg>(regCPU, 0xff), sib, lsl, offset, false), disp);
    load(w, R(result), base, disp);
    return result;
}

void JitRiscv64CodeGen::writeCPU(JitWidth w, U32 offset, RegPtr src) {
    HReg value = R(src);
    if (w == JitWidth::b8 && src->isHigh) {
        as.srli(SCR1, value, 8);
        value = SCR1;
    }
    HReg base;
    S32 disp;
    cpuAddress(offset, base, disp);
    store(w, value, base, disp);
}

void JitRiscv64CodeGen::writeCPU(JitWidth w, RegPtr sib, U8 lsl, U32 offset, RegPtr src) {
    writeHost(w, createMemPtr(std::make_shared<JitReg>(regCPU, 0xff), sib, lsl, offset, false), src, false);
}

void JitRiscv64CodeGen::writeCPUValue(JitWidth w, U32 offset, DYN_PTR_SIZE value) {
    HReg src = zero;
    if (value) {
        as.li(SCR1, (S64)value);
        src = SCR1;
    }
    HReg base;
    S32 disp;
    cpuAddress(offset, base, disp);
    store(w, src, base, disp);
}

void JitRiscv64CodeGen::writeCPUValue(JitWidth w, RegPtr sib, U8 lsl, U32 offset, DYN_PTR_SIZE value) {
    HReg src = zero;
    if (value) {
        as.li(SCR1, (S64)value);
        src = SCR1;
    }
    S32 disp;
    HReg base = hostAddress(createMemPtr(std::make_shared<JitReg>(regCPU, 0xff), sib, lsl, offset, false), disp);
    store(w, src, base, disp);
}

// ---------------------------------------------------------------------------
// host calls

void JitRiscv64CodeGen::loadCacheFromCPU() {
    for (int i = 0; i < 8; i++) {
        as.lwu(guestRegs[i], regCPU, (S32)CPU::offsetofReg32(i));
    }
}

void JitRiscv64CodeGen::writeCacheToCPU() {
    for (int i = 0; i < 8; i++) {
        as.sw(guestRegs[i], regCPU, (S32)CPU::offsetofReg32(i));
    }
}

void JitRiscv64CodeGen::setParams(const std::vector<DynParam>& params) {
    static_assert(MAX_NUMBER_OF_CALL_PARAMS == 2);
    for (std::size_t i = 0; i < params.size(); i++) {
        const DynParam& param = params[i];
        HReg dst = (HReg)(a0 + i);
        switch (param.type) {
        case JitCallParamType::REG_8:
            as.andi(dst, R(param.reg), 0xff);
            break;
        case JitCallParamType::REG_16:
            as.zext_h(dst, R(param.reg));
            break;
        case JitCallParamType::REG_32:
            as.sext_w(dst, R(param.reg)); // LP64 passes 32-bit values sign extended
            break;
        case JitCallParamType::CPU:
            as.mv(dst, regCPU);
            break;
        case JitCallParamType::CONST_8:
            as.li(dst, (S64)(param.value & 0xff));
            break;
        case JitCallParamType::CONST_16:
            as.li(dst, (S64)(param.value & 0xffff));
            break;
        case JitCallParamType::CONST_32:
            as.li(dst, (S64)(S32)(U32)param.value);
            break;
        case JitCallParamType::CONST_PTR:
            as.li(dst, (S64)param.value);
            break;
        default:
            kpanic_fmt("JitRiscv64CodeGen::setParams unknown type %d", (int)param.type);
        }
    }
}

// Temps in caller-saved registers can be live across a host call.
std::vector<U8> JitRiscv64CodeGen::spillCallerSavedTmps(U8 except) {
    std::vector<U8> spilled;
    for (int i = 0; i < NUMBER_OF_TMPS; i++) {
        U8 reg = tmpRegs[i];
        if (regUsed[reg] && reg != except && isCallerSaved(reg)) {
            as.sd((HReg)reg, regCPU, (S32)(offsetof(CPU, riscvTmpSpill) + reg * 8));
            spilled.push_back(reg);
        }
    }
    return spilled;
}

void JitRiscv64CodeGen::reloadSpilledTmps(const std::vector<U8>& spilled) {
    for (U8 reg : spilled) {
        as.ld((HReg)reg, regCPU, (S32)(offsetof(CPU, riscvTmpSpill) + reg * 8));
    }
}

void JitRiscv64CodeGen::callHostFunction(void* address, const std::vector<DynParam>& params, bool restoreCache, bool saveCache) {
    std::vector<U8> spilled = spillCallerSavedTmps(INVALID_REG);
    if (saveCache) {
        writeCacheToCPU();
    }
    setParams(params);
    as.li(SCR0, (S64)(DYN_PTR_SIZE)address);
    as.jalr(ra, SCR0, 0);
    if (restoreCache) {
        loadCacheFromCPU();
    }
    reloadSpilledTmps(spilled);
}

void JitRiscv64CodeGen::callHostFunctionWithResult(RegPtr result, void* address, const std::vector<DynParam>& params) {
    std::vector<U8> spilled = spillCallerSavedTmps(result->hardwareReg());
    writeCacheToCPU();
    setParams(params);
    as.li(SCR0, (S64)(DYN_PTR_SIZE)address);
    as.jalr(ra, SCR0, 0);
    as.mv(R(result), a0);
    loadCacheFromCPU();
    reloadSpilledTmps(spilled);
}

// ---------------------------------------------------------------------------
// code buffer and entry/exit stubs

U32 JitRiscv64CodeGen::markBufferLocation() {
    Label label = as.newLabel();
    as.bind(label);
    bufferLabels.push_back(label);
    return (U32)bufferLabels.size() - 1;
}

U32 JitRiscv64CodeGen::getBufferLocation(U32 id) {
    return as.labelOffset(bufferLabels.at(id));
}

U32 JitRiscv64CodeGen::getBufferSize() {
    if (!resolved) {
        as.resolve();
        resolved = true;
    }
    return as.size();
}

void JitRiscv64CodeGen::copyBuffer(U8* dst, U32 size) {
    memcpy(dst, as.data(), size);
}

// cpu->storedRegs layout: ra, then s0-s11.
static const HReg calleeSaved[12] = { s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11 };

// Called from C as OpCallback(CPU* cpu, DecodedOp* op): a0 = cpu, a1 = op.
U8* JitRiscv64CodeGen::createStartJITCode() {
    as.sd(ra, a0, (S32)offsetof(CPU, storedRegs));
    for (int i = 0; i < 12; i++) {
        as.sd(calleeSaved[i], a0, (S32)(offsetof(CPU, storedRegs) + 8 + i * 8));
    }
    as.mv(regCPU, a0);
    as.ld(t0, a1, (S32)offsetof(DecodedOp, pfnJitCode));
    as.li(regMMU, (S64)(DYN_PTR_SIZE)&getMemData(KThread::currentThread()->memory)->mmu);
    as.li(regEmulateSingleOp, (S64)(DYN_PTR_SIZE)cpu->thread->process->emulateSingleOp);
    loadCacheFromCPU();
    If(JitWidth::b64, std::make_shared<JitReg>(t0, 0xff)); {
        as.jr(t0);
    } StartElse(); {
        // Invalidation can clear the entry after an interpreter tail call has
        // selected this wrapper. Refetch at the current EIP in the run loop.
        writeCPUValue(DYN_PTR, offsetof(CPU, nextOp), 0);
    } EndIf();
    blockExit();
    return createDynamicExecutableMemory();
}

U8* JitRiscv64CodeGen::createBlockExit() {
    writeCacheToCPU();
    as.mv(t0, regCPU);
    as.ld(ra, t0, (S32)offsetof(CPU, storedRegs));
    for (int i = 0; i < 12; i++) {
        as.ld(calleeSaved[i], t0, (S32)(offsetof(CPU, storedRegs) + 8 + i * 8));
    }
    as.ret();
    return createDynamicExecutableMemory();
}

U8* JitRiscv64CodeGen::createSyncToHost() {
    loadCacheFromCPU();
    as.ret();
    return createDynamicExecutableMemory();
}

U8* JitRiscv64CodeGen::createSyncFromHost() {
    writeCacheToCPU();
    as.ret();
    return createDynamicExecutableMemory();
}

static void onEmitterError(const char* msg) {
    kpanic_fmt("JitRiscv64CodeGen: %s", msg);
}

JitCodeGen* startNewJIT(CPU* cpu) {
    Emitter::setErrorHandler(onEmitterError);
    return new JitRiscv64CodeGen(cpu);
}

void startNewJIT(CPU* cpu, U32 address, DecodedOp* op) {
    Emitter::setErrorHandler(onEmitterError);
    JitRiscv64CodeGen data(cpu);
    data.doJIT(address, op);
}

void clearJitBlock(const std::vector<void*>& jitOps) {
    // Without host exceptions, invalidated blocks are unlinked through
    // DecodedOp::pfnJitCode and never patched in place (same as 32-bit x86).
}

#endif
