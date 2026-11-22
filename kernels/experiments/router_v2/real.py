"""GPU-event time per router call at the real shape (fp32 [128, 2304], 48 distinct banks -> DRAM), K=8."""
import sys, statistics, torch
sys.path.insert(0, "/Users/arthurzucker/Work/hub-kernels/metal-moe-decode/build/torch-metal")
import router as old
sys.path.insert(0, "."); del sys.modules["router"]; import router as new
torch.manual_seed(0)
banks = [(torch.randn(128, 2304, device="mps") * 0.02, torch.randn(128, device="mps") * 0.01) for _ in range(48)]
x = torch.randn(1, 2304, device="mps", dtype=torch.bfloat16)
cnt = torch.zeros(1, dtype=torch.int32, device="mps")
lo, ow, oi = torch.empty(1, 128, device="mps"), torch.empty(1, 8, device="mps"), torch.empty(1, 8, dtype=torch.int64, device="mps")
cfgs = [("old", getattr(old._lib(), "router_topk_bfloat"), 16)] + [
    (f"new SPLIT={s}", getattr(new._lib(), f"router_topk_bfloat_{s}"), 128 // (8 // s)) for s in (1, 2, 4, 8)]
for name, k, groups in cfgs:
    fn = lambda: [k(x, w, b, lo, ow, oi, cnt, 128, 2304, 8, 2.5, threads=groups * 256, group_size=256) for w, b in banks]
    for _ in range(3): fn()
    res = []
    for _ in range(7):
        torch.mps.synchronize(); s, e = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize(); res.append(s.elapsed_time(e) / 48 * 1e3)
    us = statistics.median(res)
    print(f"{name:12s} {us:5.1f} us/call  {128*2304*4/us/1e3:4.0f} GB/s", flush=True)
