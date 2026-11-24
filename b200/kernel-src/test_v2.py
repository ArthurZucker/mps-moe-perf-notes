"""v2 vs v1 expert kernels at Qwen3.8 shapes: exactness (resident + streamed) and timing under CUDA graphs over 48
distinct layer banks (HBM, not L2)."""
import torch, sys
from pathlib import Path
from kernels import get_local_kernel
k = get_local_kernel(Path(sys.argv[1]))
torch.manual_seed(0)
E, H, I, topk, G, NL = 512, 2560, 640, 10, 64, 8
def bank(N, K, dev="cuda", n=E):
    w = torch.randint(-2**31, 2**31 - 1, (n, N, K // 8), dtype=torch.int32, device=dev)
    s = (torch.rand(n, N, K // G, device=dev) * 0.02).bfloat16(); b = (torch.randn(n, N, K // G, device=dev) * 0.05).bfloat16()
    return w, s, b
layers = [(bank(2 * I, H), bank(H, I)) for _ in range(NL)]
x = torch.randn(1, H, device="cuda").bfloat16()
act = (torch.randn(topk, I, device="cuda") * 0.5).bfloat16()
ids = torch.randperm(E, device="cuda")[:topk].int(); wts = torch.rand(topk, device="cuda")
gu, dn = layers[0]
# exactness resident
a = k.gather_qmv4(x, *gu, ids, topk, G); b = k.qmv4_v2(x, gu, ids, 1, topk, G, rows=4)
c = k.down_route4(act, *dn, ids, wts, topk, G); d = k.qmv4_v2(act, dn, ids, topk, 1, G, weights=wts, rows=1)
print("resident gate_up equal:", torch.equal(a, b), " down equal:", torch.equal(c, d))
# streamed exactness: pool of 16 slots, host banks pinned
S = 16
host_gu = [t.cpu().pin_memory() for t in gu]; host_dn = [t.cpu().pin_memory() for t in dn]
pool_gu = [torch.zeros(S, *t.shape[1:], dtype=t.dtype, device="cuda") for t in gu]
pool_dn = [torch.zeros(S, *t.shape[1:], dtype=t.dtype, device="cuda") for t in dn]
table = torch.full((E,), -1, dtype=torch.int32, device="cuda"); owner = torch.full((S,), -1, dtype=torch.int32, device="cuda")
age = torch.full((S,), -1, dtype=torch.long, device="cuda"); clock = torch.zeros(1, dtype=torch.long, device="cuda")
ok = True
for step in range(20):
    ids2 = torch.randperm(24, device="cuda")[:topk].int()
    slots, miss = k.offload_plan(ids2, 0, E, table, owner, age, clock)
    g2 = k.qmv4_v2(x, pool_gu, ids2, 1, topk, G, host=[t.data_ptr() for t in host_gu], slots=slots, miss=miss, rows=4)
    d2 = k.qmv4_v2(act, pool_dn, ids2, topk, 1, G, host=[t.data_ptr() for t in host_dn], slots=slots, miss=miss, weights=wts, rows=1)
    ok &= torch.equal(g2, k.gather_qmv4(x, *gu, ids2, topk, G)) and torch.equal(d2, k.down_route4(act, *dn, ids2, wts, topk, G))
print("streamed v2 == resident v1 over 20 steps (with installs):", ok)
def gt(fns, it=50):
    def f():
        for fn in fns: fn()
    f(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    g.replay(); torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(it): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it / len(fns) * 1e3
by_gu = topk * gu[0][0].numel() * 4 * 1.125; by_dn = topk * dn[0][0].numel() * 4 * 1.125
print(f"v1 gate_up {gt([lambda L=L: k.gather_qmv4(x, *L[0], ids, topk, G) for L in layers]):6.1f} us   down {gt([lambda L=L: k.down_route4(act, *L[1], ids, wts, topk, G) for L in layers]):6.1f} us")
for rows in (4, 2, 1):
    t1 = gt([lambda L=L: k.qmv4_v2(x, L[0], ids, 1, topk, G, rows=rows) for L in layers])
    print(f"v2 gate_up rows={rows} {t1:6.1f} us ({by_gu/t1/1e3:.0f} GB/s)")
for rows in (2, 1):
    t2 = gt([lambda L=L: k.qmv4_v2(act, L[1], ids, topk, 1, G, weights=wts, rows=rows) for L in layers])
    print(f"v2 down    rows={rows} {t2:6.1f} us ({by_dn/t2/1e3:.0f} GB/s)")
