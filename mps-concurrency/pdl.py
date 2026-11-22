import os, statistics, torch
from torch.utils.cpp_extension import load
here = os.path.dirname(os.path.abspath(__file__))
ext = load("_mps_concurrency", [f"{here}/mps_concurrency.mm"], build_directory=f"{here}/.build",
           extra_cflags=["-std=c++20", "-O2", "-D_DARWIN_C_SOURCE", "-include", "sys/types.h", "-fobjc-arc"],
           extra_ldflags=["-framework", "Metal", "-framework", "Foundation"], verbose=False)
torch.zeros(1, device="mps")
n = 960
for tgs in (2, 8, 32, 128):
    res = {}
    for mode in (0, 1):
        for _ in range(2): ext.chain_pdl(n, mode, tgs)
        runs = [ext.chain_pdl(n, mode, tgs) for _ in range(7)]
        res[mode] = statistics.median(r[0] for r in runs); ok = all(r[1] == 1.0 for r in runs)
    kb = tgs * 8 * 2304 * 2 / 1e3
    print(f"{tgs * 8:5d} rows ({kb:6.0f} KB weights): serial {res[0] / n:6.2f} us/step | software PDL {res[1] / n:6.2f} us/step | x{res[0] / res[1]:.2f} | complete={ok}", flush=True)
