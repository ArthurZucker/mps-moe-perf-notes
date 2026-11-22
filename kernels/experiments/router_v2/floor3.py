import sys, time, statistics, torch
sys.path.insert(0, "/Users/arthurzucker/Work/hub-kernels/metal-moe-decode/build/torch-metal")
import router as R
def timed(fn, n=48):
    for _ in range(5): fn()
    ts = []
    for _ in range(9):
        torch.mps.synchronize(); t = time.perf_counter(); fn(); torch.mps.synchronize(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) / n * 1e6
for H, K in ((64, 1), (64, 8), (2304, 1), (2304, 8)):
    w = torch.randn(8, H, device="mps"); b = torch.zeros(8, device="mps"); x = torch.randn(1, H, device="mps", dtype=torch.bfloat16)
    print(f"E=8 H={H:5d} K={K}: op {timed(lambda: [torch.ops.metal_router.router_topk(x, w, b, K, 1.0) for _ in range(48)]):5.1f} us"
          f" | direct _router_mps {timed(lambda: [R._router_mps(x, w, b, K, 1.0) for _ in range(48)]):5.1f} us", flush=True)
