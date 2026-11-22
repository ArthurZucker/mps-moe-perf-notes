import sys, statistics, torch
sys.path.insert(0, "."); import qmv4 as Q
K = 2304
def mk(N): return (torch.randint(0, 2**31, (N, K // 8), dtype=torch.int32, device="mps").view(torch.uint32), torch.rand(N, K // 64, device="mps", dtype=torch.bfloat16) * 0.01, torch.rand(N, K // 64, device="mps", dtype=torch.bfloat16) * 0.01 - 0.005)
sets = [(mk(6552), mk(1024), mk(1024)) for _ in range(24)]
xs = [torch.randn(1, K, device="mps", dtype=torch.bfloat16) for _ in range(3)]
q, k, v = sets[0]
r = [Q.qmv4(x, *w, 64) for x, w in zip(xs, (q, k, v))]; m = Q.qmv4_multi(*xs[0:1], *q, xs[1], *k, xs[2], *v, 64)
print("max abs diff:", max((a.float() - b.float()).abs().max().item() for a, b in zip(r, m)))
def t(fn):
    for _ in range(3): fn()
    res = []
    for _ in range(7):
        torch.mps.synchronize(); a, e = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
        a.record(); fn(); e.record(); e.synchronize(); res.append(a.elapsed_time(e) / 24 * 1e3)
    return statistics.median(res)
sep = t(lambda: [[Q.qmv4(x, *w, 64) for x, w in zip(xs, s)] for s in sets])
mul = t(lambda: [Q.qmv4_multi(xs[0], *s[0], xs[1], *s[1], xs[2], *s[2], 64) for s in sets])
print(f"q+k+v separate {sep:.1f} us | multi {mul:.1f} us | saves {sep - mul:.1f} us per layer pass")
