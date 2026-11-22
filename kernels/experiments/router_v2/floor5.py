import sys, time, torch
sys.path.insert(0, "/Users/arthurzucker/Work/hub-kernels/metal-moe-decode/build/torch-metal")
import router as old
sys.path.insert(0, "."); del sys.modules["router"]; import router as new
w = torch.randn(8, 64, device="mps"); b = torch.zeros(8, device="mps"); x = torch.randn(1, 64, device="mps", dtype=torch.bfloat16)
cnt = torch.zeros(1, dtype=torch.int32, device="mps")
lo, ow, oi = torch.empty(1, 8, device="mps"), torch.empty(1, 8, device="mps"), torch.empty(1, 8, dtype=torch.int64, device="mps")
for name, k in (("old", getattr(old._lib(), "router_topk_bfloat")), ("new", getattr(new._lib(), "router_topk_bfloat_1"))):
    for K in (1, 8):
        fn = lambda: k(x, w, b, lo, ow, oi, cnt, 8, 64, K, 1.0, threads=256, group_size=256)
        for _ in range(50): fn()
        torch.mps.synchronize(); s, e = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
        s.record()
        for _ in range(480): fn()
        e.record(); e.synchronize()
        print(f"{name} K={K} E=8 H=64: gpu {s.elapsed_time(e)/480*1e3:5.1f} us", flush=True)
