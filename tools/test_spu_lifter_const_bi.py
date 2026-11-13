#!/usr/bin/env python3
"""`bi $rN` with a constant target loaded in the same block is a direct transfer.

Sony's SPU code opens and closes interrupt windows with `ila $rN,<next> ;
bid/bie $rN`. compute_const_bi_targets finds those sites and the lifter emits
SPU_DIRECT instead of a trip through the drain loop and spu_indirect_branch.
The cases below pin what qualifies and what keeps the full dispatch.

Usage:
    python3 tools/test_spu_lifter_const_bi.py
"""

import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from spu_disasm import disassemble_spu  # noqa: E402
from spu_lifter import SPULifter, compute_const_bi_targets  # noqa: E402

E_BIT, D_BIT = 0x40000, 0x80000


def ri18(op7, i18, rt):
    return (op7 << 25) | ((i18 & 0x3FFFF) << 7) | rt


def ri16(op9, i16, rt):
    return (op9 << 23) | ((i16 & 0xFFFF) << 7) | rt


def ri10(op8, i10, ra, rt):
    return (op8 << 24) | ((i10 & 0x3FF) << 14) | (ra << 7) | rt


def rr(op11, rb, ra, rt):
    return (op11 << 21) | (rb << 14) | (ra << 7) | rt


def ila(rt, k): return ri18(0x21, k, rt)
def il(rt, k): return ri16(0x081, k, rt)
def bi(ra, bits=0): return rr(0x1A8, 0, ra, 0) | bits
def lqd(rt, ra, i10): return ri10(0x34, i10, ra, rt)
def nop(): return 0x40200000
def br(src, dst): return ri16(0x064, (dst - src) >> 2, 0)
def brsl(src, dst, rt=0): return ri16(0x066, (dst - src) >> 2, rt)
def ai(rt, ra, i): return ri10(0x1C, i, ra, rt)


def assemble(words, base):
    data = b"".join(w.to_bytes(4, "big") for w in words)
    return disassemble_spu(data, base)


def classify(words, bounds, base=0x4000, exclude=()):
    insns = assemble(words, base)
    starts = {s for s, _ in bounds}
    return insns, compute_const_bi_targets(insns, bounds, starts, exclude)


def lift(words, bounds, base=0x4000):
    insns, const_bi = classify(words, bounds, base)
    lifter = SPULifter(prefix="t_")
    lifter.func_starts = {s for s, _ in bounds}
    lifter.const_bi = const_bi
    out = {}
    for s, e in bounds:
        out[s] = "\n".join(lifter.lift_function(insns, s, e).body_lines)
    return out


class ConstBi(unittest.TestCase):
    def test_bid_window(self):
        # 0x4000 ila $2,0x4008 ; 0x4004 bid $2 ; 0x4008 nop ; bi $0
        w = [ila(2, 0x4008), bi(2, D_BIT), nop(), bi(0)]
        b = [(0x4000, 0x4008), (0x4008, 0x4010)]
        self.assertEqual(classify(w, b)[1], {0x4004: 0x4008})
        body = lift(w, b)[0x4000]
        self.assertIn("ctx->int_enable = 0;", body)
        self.assertIn("SPU_DIRECT(ctx, 0x4008, t_spu_func_00004008)", body)
        self.assertNotIn("irq_saved", body)
        self.assertNotIn("spu_indirect_branch", body)

    def test_bie_keeps_irq_guard(self):
        w = [ila(2, 0x4008), bi(2, E_BIT), nop(), bi(0)]
        b = [(0x4000, 0x4008), (0x4008, 0x4010)]
        body = lift(w, b)[0x4000]
        self.assertIn("ctx->int_enable = 1;", body)
        self.assertIn("if (ctx->irq_saved)", body)
        self.assertIn("SPU_DIRECT(ctx, 0x4008, t_spu_func_00004008)", body)

    def test_il_and_identity_move(self):
        # il, then an identity move of the same register, then the branch
        w = [il(2, 0x400C), ai(2, 2, 0), bi(2), nop(), bi(0)]
        b = [(0x4000, 0x400C), (0x400C, 0x4014)]
        self.assertEqual(classify(w, b)[1], {0x4008: 0x400C})

    def test_loaded_target_stays_indirect(self):
        w = [lqd(2, 1, 0), bi(2), nop(), bi(0)]
        b = [(0x4000, 0x4008), (0x4008, 0x4010)]
        self.assertEqual(classify(w, b)[1], {})
        self.assertIn("spu_indirect_branch", lift(w, b)[0x4000])

    def test_overwritten_after_ila(self):
        w = [ila(2, 0x400C), lqd(2, 1, 0), bi(2), nop(), bi(0)]
        b = [(0x4000, 0x400C), (0x400C, 0x4014)]
        self.assertEqual(classify(w, b)[1], {})

    def test_target_not_an_entry(self):
        w = [ila(2, 0x400C), bi(2), nop(), nop(), bi(0)]
        b = [(0x4000, 0x4008), (0x4008, 0x4014)]      # 0x400C starts nothing
        self.assertEqual(classify(w, b)[1], {})

    def test_branch_lands_between(self):
        # 0x4008 br 0x4004 enters the block between the ila and the bi
        w = [ila(2, 0x400C), bi(2), br(0x4008, 0x4004), nop(), bi(0)]
        b = [(0x4000, 0x4008), (0x4008, 0x400C), (0x400C, 0x4014)]
        self.assertEqual(classify(w, b)[1], {})
        w = [ila(2, 0x4010), nop(), bi(2), br(0x400C, 0x4004), nop(), bi(0)]
        b = [(0x4000, 0x400C), (0x400C, 0x4010), (0x4010, 0x4018)]
        self.assertEqual(classify(w, b)[1], {})

    def test_call_between(self):
        w = [ila(2, 0x400C), brsl(0x4004, 0x4010), bi(2), nop(), bi(0)]
        b = [(0x4000, 0x400C), (0x400C, 0x4010), (0x4010, 0x4014)]
        self.assertEqual(classify(w, b)[1], {})

    def test_special_and_cross_task_lo(self):
        # 0xA70 is the taskset syscall address spu_indirect_branch intercepts
        base = 0xA60
        w = [ila(2, 0xA70), bi(2), nop(), nop(), nop(), bi(0)]
        b = [(0xA60, 0xA70), (0xA70, 0xA78)]
        self.assertEqual(classify(w, b, base)[1], {})
        # policy-side code branching up into the task area keeps the dispatch
        base = 0x2FF8
        w = [ila(2, 0x3000), bi(2), nop(), bi(0)]
        b = [(0x2FF8, 0x3000), (0x3000, 0x3008)]
        self.assertEqual(classify(w, b, base)[1], {})

    def test_excluded_site(self):
        w = [ila(2, 0x4008), bi(2), nop(), bi(0)]
        b = [(0x4000, 0x4008), (0x4008, 0x4010)]
        self.assertEqual(classify(w, b, exclude={0x4004})[1], {})


if __name__ == "__main__":
    unittest.main()
