#!/usr/bin/env python3
"""SPU_DMA_DEFER: deferred GET copies and their flush points (test_spu_dma_defer.c)."""
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

if os.name == "nt":
    print("SKIP: DMA defer fixture currently requires a POSIX C compiler")
    raise SystemExit(0)
root = Path(__file__).resolve().parents[1]
spu = root / "runtime/spu"
common = ["spu_drain.c", "spu_lockstep.c", "spu_coherency.c", "spu_workload.c",
          "spurs_policy.c", "spurs_job.c", "spu_tsp_weak.c", "spu_vm_pagemap.c",
          "spurs_policy_blob_weak.c", "spu_interp.c"]
env = {k: v for k, v in os.environ.items()
       if not k.startswith(("SPU_", "SPURS_", "YDKJ_", "LBP_", "GH3_"))}


def run(binary, cwd, *args, extra=None):
    p = subprocess.run([str(binary), *args], env={**env, **(extra or {})},
                       cwd=cwd, capture_output=True, text=True, timeout=30)
    print(p.stdout, end="")
    if p.returncode:
        print(p.stderr)
        raise SystemExit(p.returncode)
    return p.stderr


with tempfile.TemporaryDirectory(prefix="spu-dma-defer-") as tmp:
    binary = Path(tmp) / "fixture"
    subprocess.run([os.environ.get("CC", "cc"), "-std=gnu17", "-O1", "-UNDEBUG",
                    *shlex.split(os.environ.get("CFLAGS", "")), "-pthread",
                    "-I" + str(root / "include"), "-I" + str(spu),
                    "-I" + str(root / "runtime/platform"),
                    str(spu / "tests/test_spu_dma_defer.c"),
                    *[str(spu / name) for name in common],
                    str(root / "runtime/platform/win32_compat.c"), "-lm",
                    "-o", str(binary)], check=True)
    err = run(binary, tmp, extra={"SPU_DMA_DEFER": "1"})
    if "SPU_DMA_DEFER=1" not in err:
        print(err)
        raise SystemExit("the fixture did not report SPU_DMA_DEFER=1")
    run(binary, tmp, "ring", extra={"SPU_DMA_DEFER": "1", "SPU_DGQ_RING": "1"})
    run(binary, tmp, "off")
