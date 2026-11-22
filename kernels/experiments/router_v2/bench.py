"""router_topk v2 (rows split over SPLIT simdgroups) vs the published kernel: correctness + GPU time per call over
48 distinct fp32 [128, 2304] banks (DRAM reads), through the GPU lock."""
import sys, time, statistics, torch
sys.path.insert(0, "/Users/arthurzucker/Work/hub-kernels/metal-moe-decode/build/torch-metal")
import router as old  # noqa: F401  registers metal_router
import router as _o
sys.path.insert(0, "."); del sys.modules["router"]
import router as new
E_, H, K = 128, 2304, 8
torch.manual_seed(0)
banks = [(torch.randn(E_, H, device="mps") * 0.02, torch.randn(E_, device="mps") * 0.01) for _ in range(48)]
xs = [torch.randn(1, H, device="mps", dtype=torch.bfloat16) for _ in range(4)]
ref_op, new_op = torch.ops.metal_router.router_topk, torch.ops.metal_router_v2.router_topk

def timed(fn):
    for _ in range(5): fn()
    ts = []
    for _ in range(9):
        torch.mps.synchronize(); t = time.perf_counter(); fn(); torch.mps.synchronize(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) / 48 * 1e6

base = timed(lambda: [ref_op(xs[0], w, b, K, 2.5) for w, b in banks])
print(f"published        {base:6.1f} us/call {E_*H*4/base/1e3:5.0f} GB/s", flush=True)
for split in (8,):
    new.SPLIT = split
    bad = 0
    for x in xs:
        for w, b in banks[:12]:
            l0, w0, i0 = ref_op(x, w, b, K, 2.5); l1, w1, i1 = new_op(x, w, b, K, 2.5)
            same = set(i0[0].tolist()) == set(i1[0].tolist())
            bad += (not same) or (l0 - l1).abs().max().item() > 1e-4
    us = timed(lambda: [new_op(xs[0], w, b, K, 2.5) for w, b in banks])
    print(f"v2 SPLIT={split}       {us:6.1f} us/call {E_*H*4/us/1e3:5.0f} GB/s  mismatches {bad}/48", flush=True)
