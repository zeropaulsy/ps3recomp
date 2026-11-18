#!/usr/bin/env python3
"""Rewrite lifted SPU C so that reads of registers written long ago load from ctx.

The lifted code stores every SPU result to ctx->gpr[N] and reads operands back from
there. GCC forwards each store to the later reads and keeps the value in an x86
register; with 128 SPU registers and 16 xmm, most of those values are then spilled to
the stack and reloaded, though ctx->gpr[N] already holds them. In long straight-line
job functions about a third of the code was such spills (jobmod_5090d4879c02
func_8000: 1754 stack instructions of ~6000), and that code is instruction-fetch bound.

A read of ctx->gpr[N] whose last write, in the same function and since the last label,
is more than K writes back becomes SPU_FAR(...):
an opaque load (spu_helpers.h) that GCC cannot replace with the stored value, so the
register is read from ctx and no spill is made. A read of an integer member
(ctx->gpr[N]._u32[0], for addresses and branch conditions) becomes SPU_FAR_S(...), a
plain integer load (a whole-register load plus a move made LBP2's lift 4% larger).
The result is the same for any K; K only steers code size.
Microbench of func_8000 on a captured entry state: K=8 631 ns/call vs 746 (-15%);
K=0 654, K=4 651, K=12 639.

usage: spu_far_reads.py [-k K] file.c [...]    rewrite in place (skips rewritten files)
       spu_far_reads.py [-k K] -o out.c in.c   write to out.c
"""
import argparse
import re
import sys

MARK = "/* spu_far_reads K="
# ctx->gpr[N], optionally followed by a member access ._name[k].
ACCESS = re.compile(r"(&\s*)?ctx->gpr\[(\d+)\](\._(\w+)\[\d+\])?")
# What follows an access that makes it a write (plain or compound assignment).
WRITE_AFTER = re.compile(r"\s*(?:[-+*/%|&^]|<<|>>)?=(?!=)")
SCALAR = {"u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64"}
FUNC_START = re.compile(r"^(?:static\s+)?(?:inline\s+)?void\s+\w+\s*\(")
LABEL = re.compile(r"^\s*\w+:\s*$")


def transform(lines, k):
    out = []
    last = {}       # register -> write counter value at its last write
    n = 0           # writes seen so far in the current function
    for line in lines:
        if FUNC_START.match(line) or LABEL.match(line):
            last = {}
            out.append(line)
            continue
        if "ctx->gpr[" not in line:
            out.append(line)
            continue
        pieces, pos, writes = [], 0, []
        for m in ACCESS.finditer(line):
            reg = int(m.group(2))
            if m.group(1) or WRITE_AFTER.match(line, m.end()):
                if not m.group(3):
                    writes.append(reg)
                continue
            if reg not in last or n - last[reg] <= k:
                continue
            if m.group(3) and m.group(4) in SCALAR:
                rep = f"SPU_FAR_S(ctx->gpr[{reg}]{m.group(3)})"
            else:
                rep = f"SPU_FAR(ctx->gpr[{reg}]){m.group(3) or ''}"
            pieces.append(line[pos:m.start()])
            pieces.append(rep)
            pos = m.end()
        pieces.append(line[pos:])
        out.append("".join(pieces))
        for reg in writes:
            n += 1
            last[reg] = n
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-k", type=int, default=8, help="max write distance kept in registers")
    ap.add_argument("-o", help="output file (single input)")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.o and len(a.files) != 1:
        sys.exit("-o takes exactly one input")
    for path in a.files:
        src = open(path).read()
        if MARK in src:
            if a.o:
                open(a.o, "w").write(src)
            else:
                print(f"skip (already rewritten): {path}", file=sys.stderr)
            continue
        lines = transform(src.split("\n"), a.k)
        text = f"{MARK}{a.k} (tools/spu_far_reads.py) */\n" + "\n".join(lines)
        with open(a.o or path, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
