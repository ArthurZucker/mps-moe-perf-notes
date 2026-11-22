"""Serial vs concurrent compute encoders on Apple GPUs at decode-like kernel sizes (see mps_concurrency.mm)."""
import os, statistics, torch
from torch.utils.cpp_extension import load
here = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f"{here}/.build", exist_ok=True)
ext = load("_mps_concurrency", [f"{here}/mps_concurrency.mm"], build_directory=f"{here}/.build",
           extra_cflags=["-std=c++20", "-O2", "-D_DARWIN_C_SOURCE", "-include", "sys/types.h", "-fobjc-arc"],
           extra_ldflags=["-framework", "Metal", "-framework", "Foundation"], verbose=False)
torch.zeros(1, device="mps")
N = 1200
MODES = ["serial-chain", "conc-chain", "serial-indep", "conc-indep", "encoder-per-op"]
def med(mode, kind, group=1):
    for _ in range(3): ext.run(mode, N, group, kind)
    return statistics.median(ext.run(mode, N, group, kind) for _ in range(9))
for kind, name in ((0, "tiny (4096 threads)"), (1, "matvec 1024x2304 bf16")):
    print(f"== {name}, {N} dispatches per command buffer")
    for mode in (0, 1, 4):
        us = med(mode, kind); print(f"   {MODES[mode]:15s} {us / 1e3:7.2f} ms  ({us / N:5.2f} us/dispatch)", flush=True)
    for g in (2, 3, 4, 8):
        s, c = med(2, kind, g), med(3, kind, g)
        print(f"   independent groups of {g}: serial {s / N:5.2f} us/dispatch | concurrent {c / N:5.2f} us/dispatch | x{s / c:4.2f}", flush=True)
