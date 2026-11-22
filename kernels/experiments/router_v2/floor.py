import sys, time, statistics, torch
sys.path.insert(0, "/Users/arthurzucker/Work/hub-kernels/metal-moe-decode/build/torch-metal")
import router  # noqa
op = torch.ops.metal_router.router_topk
def timed(fn, n):
    for _ in range(5): fn()
    ts = []
    for _ in range(9):
        torch.mps.synchronize(); t = time.perf_counter(); fn(); torch.mps.synchronize(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) / n * 1e6
for E_ in (8, 32, 128):
    w = [torch.randn(E_, 2304, device="mps") for _ in range(48)]; b = torch.zeros(E_, device="mps")
    x = torch.randn(1, 2304, device="mps", dtype=torch.bfloat16)
    print(f"E={E_:4d} {timed(lambda: [op(x, wi, b, 8, 1.0) for wi in w], 48):6.1f} us/call", flush=True)
x = torch.randn(2304, device="mps")
print(f"tiny add (eager) {timed(lambda: [x.add_(1) for _ in range(48)], 48):6.1f} us/call")
