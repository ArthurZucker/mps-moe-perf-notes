import os, statistics, torch
from torch.utils.cpp_extension import load
here = os.path.dirname(os.path.abspath(__file__))
ext = load("_mps_concurrency", [f"{here}/mps_concurrency.mm"], build_directory=f"{here}/.build",
           extra_cflags=["-std=c++20", "-O2", "-D_DARWIN_C_SOURCE", "-include", "sys/types.h", "-fobjc-arc"],
           extra_ldflags=["-framework", "Metal", "-framework", "Foundation"], verbose=False)
torch.zeros(1, device="mps")
N = 960  # 10 passes over the 96-weight ring
def med(ahead, th=4096):
    for _ in range(2): ext.chain_prefetch(N, ahead, th)
    return statistics.median(ext.chain_prefetch(N, ahead, th) for _ in range(7))
base = med(0)
mb = 1024 * 2304 * 2 / 1e6
print(f"DRAM-bound chain, {N} dependent matvecs of 1024x2304 bf16 ({mb:.1f} MB each): {base / N:6.2f} us/step ({mb * 1e3 / (base / N):.0f} GB/s)", flush=True)
for ahead in (1, 2, 4, 8):
    for th in (4096, 16384):
        us = med(ahead, th)
        print(f"   + prefetch queue {ahead} steps ahead ({th} threads): {us / N:6.2f} us/step  x{base / us:4.2f}", flush=True)
