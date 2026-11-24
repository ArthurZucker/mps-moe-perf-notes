import torch, sys, time
from pathlib import Path
from kernels import get_local_kernel
k = get_local_kernel(Path(sys.argv[1]))
nthreads = int(sys.argv[2]) if len(sys.argv) > 2 else 32
torch.manual_seed(0)
E, H, I, G, topk, L = 64, 2560, 640, 64, 10, 2
def bank(N, K):
    w = torch.randint(-2**31, 2**31 - 1, (E, N, K // 8), dtype=torch.int32)
    s = (torch.rand(E, N, K // G) * 0.02).bfloat16(); b = (torch.randn(E, N, K // G) * 0.05).bfloat16()
    return [t.pin_memory() for t in (w, s, b)]
host = [(bank(2 * I, H), bank(H, I)) for _ in range(L)]
dev = [tuple(tuple(t.cuda() for t in b) for b in hb) for hb in host]
ptrs = [t.data_ptr() for l in host for b in l for t in b]
cpus = [int(c) for c in __import__("os").environ.get("CPUS", ",".join(map(str, range(8, 8 + nthreads)))).split(",")]
handle, mb, res = k.host_tier_start(ptrs, H, I, G, cpus)
state = torch.zeros(2, dtype=torch.long, device="cuda")
table = torch.full((L * E,), -1, dtype=torch.int32, device="cuda")
# make experts 0..31 of each layer "resident" (slot = expert id; the GPU banks here are full banks, so slot == id)
for l in range(L): table[l * E: l * E + 32] = torch.arange(32, dtype=torch.int32, device="cuda")
def moe_gpu_ref(x, ids, w, l):
    gu = k.gather_qmv4(x, *dev[l][0], ids, topk, G)
    act = (torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]).contiguous()
    return k.down_route4(act, *dev[l][1], ids, w, topk, G)
def moe_split(x, ids, w, l):
    resident = table[l * E + ids.long()] >= 0
    gw = torch.where(resident, w, torch.zeros_like(w)).contiguous()
    k.host_post(ids, w, l, E, table, x, mb, state)
    gu = k.gather_qmv4(x, *dev[l][0], ids, topk, G)
    act = (torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]).contiguous()
    out = k.down_route4(act, *dev[l][1], ids, gw, topk, G)
    return k.host_wait(out, res, state)
worst = 0
for step in range(20):
    l = step % L
    x = torch.randn(1, H, device="cuda").bfloat16()
    ids = torch.randperm(E, device="cuda")[:topk].int(); w = torch.rand(topk, device="cuda")
    ref = moe_gpu_ref(x, ids, w, l).float(); got = moe_split(x, ids, w, l).float()
    worst = max(worst, ((got - ref).norm() / ref.norm()).item())
print(f"host tier vs GPU (rel err, max over 20 calls with ~5 misses each): {worst:.2e}; host calls {k.host_tier_calls(handle)}")
# round-trip latency of post -> wait with n misses, in a CUDA graph (the GPU work beside it is the resident path)
for n in (0, 1, 2, 4, 10):
    ids = torch.cat([torch.arange(10 - n), torch.arange(32, 32 + n)]).int().cuda(); w = torch.rand(topk, device="cuda")
    x = torch.randn(1, H, device="cuda").bfloat16()
    f = lambda: moe_split(x, ids, w, 0)
    f(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(8): f()
    g.replay(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20): g.replay()
    torch.cuda.synchronize(); us = (time.perf_counter() - t) / 160 * 1e6
    print(f"{n:2d} missed experts: {us:6.1f} us per MoE call (post + GPU hits + host compute + wait)")
# fused persistent layer kernel vs the split path
age = torch.full((E,), -1, dtype=torch.long, device="cuda"); clock = torch.zeros(1, dtype=torch.long, device="cuda")
def moe_fused(x, ids, w, l):
    return k.moe_layer_host(x, ids, w, l, E, table, age, clock, [*dev[l][0], *dev[l][1]], mb, res, state, G, 148, True)
def moe_fused_post(x, ids, w, l):
    k.host_post(ids, w, l, E, table, x, mb, state)
    return k.moe_layer_host(x, ids, w, l, E, table, age, clock, [*dev[l][0], *dev[l][1]], mb, res, state, G, 148, False)
worst = 0
for step in range(20):
    l = step % L
    x = torch.randn(1, H, device="cuda").bfloat16()
    ids = torch.randperm(E, device="cuda")[:topk].int(); w = torch.rand(topk, device="cuda")
    a = moe_split(x, ids, w, l).float(); b = moe_fused(x, ids, w, l).float()
    worst = max(worst, ((a - b).norm() / a.norm()).item())
print(f"fused vs split rel diff (max of 20): {worst:.2e}")
for n in (0, 2, 4, 10):
    ids = torch.cat([torch.arange(10 - n), torch.arange(32, 32 + n)]).int().cuda(); w = torch.rand(topk, device="cuda")
    x = torch.randn(1, H, device="cuda").bfloat16()
    for name, fn in (("split", moe_split), ("fused", moe_fused), ("post+fused", moe_fused_post)):
        f = lambda: fn(x, ids, w, 0)
        f(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(8): f()
        g.replay(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20): g.replay()
        torch.cuda.synchronize(); us = (time.perf_counter() - t) / 160 * 1e6
        print(f"{n:2d} misses {name}: {us:6.1f} us/layer")
def fused_nohost(x, ids, w, l):
    return k.moe_layer_host(x, ids, w, l, E, table, age, clock, [*dev[l][0], *dev[l][1]], 0, 0, state, G, 148, True)
z = torch.zeros(1, H, device="cuda", dtype=torch.bfloat16)
def host_only(x, ids, w, l):
    k.host_post(ids, w, l, E, table, x, mb, state)
    return k.host_wait(z, res, state)
for n in (2, 10):
    ids = torch.cat([torch.arange(10 - n), torch.arange(32, 32 + n)]).int().cuda(); w = torch.rand(topk, device="cuda")
    x = torch.randn(1, H, device="cuda").bfloat16()
    for name, fn in (("fused no-host", fused_nohost), ("host round trip only", host_only), ("split", moe_split)):
        f = lambda: fn(x, ids, w, 0)
        f(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(8): f()
        g.replay(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20): g.replay()
        torch.cuda.synchronize(); us = (time.perf_counter() - t) / 160 * 1e6
        tm = k.host_tier_timing(handle)
        cpu = f"  [CPU: gate/up {tm[0] / max(tm[2], 1) / 1e3:.1f} us, down {tm[1] / max(tm[2], 1) / 1e3:.1f} us per call]" if tm[2] else ""
        print(f"DIAG {n:2d} misses {name}: {us:6.1f} us{cpu}")
k.host_tier_stop(handle)
