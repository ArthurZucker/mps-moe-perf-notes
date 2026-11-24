import os, torch
from kernels import get_local_kernel
from pathlib import Path
k = get_local_kernel(Path(os.environ["TRANSFORMERS_MOE_AFFINE_KERNEL"]))
E, H, I, TOPK, G, S = 512, 2560, 640, 10, 64, 128
dev = "cuda"; torch.manual_seed(0)
def bank(N, K, E, pin=False):
    w = torch.randint(-2**31, 2**31 - 1, (E, N, K // 8), dtype=torch.int32)
    s = (torch.randn(E, N, K // G) * 0.01).bfloat16(); b = (torch.randn(E, N, K // G) * 0.01).bfloat16()
    return tuple(t.pin_memory() if pin else t.to(dev) for t in (w, s, b))
gu, dn = bank(2 * I, H, E), bank(H, I, E)
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
def timeit(f, n=200):
    for _ in range(3): f()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): f()
    g.replay(); torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1e3 / n
for T in (1, 2, 3, 4):
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)]).reshape(-1).int()
    x = torch.randn(T, H, device=dev).bfloat16(); a = torch.randn(T * TOPK, I, device=dev).bfloat16()
    w = torch.rand(T * TOPK, device=dev)
    r1 = k.gather_qmv4(x, *gu, ids, TOPK, G); n1 = k.qmv4_tc6(x, gu, ids, 1, TOPK, G)
    r2 = k.down_route4(a, *dn, ids, w, TOPK, G); n2 = k.qmv4_tc6(a, dn, ids, TOPK, 1, G, weights=w)
    idss = [torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)]).reshape(-1).int() for _ in range(200)]
    it = iter(range(10**9))
    t_old = timeit(lambda: k.gather_qmv4(x, *gu, idss[next(it) % 200], TOPK, G)) + timeit(lambda: k.down_route4(a, *dn, idss[next(it) % 200], w, TOPK, G))
    t_new = timeit(lambda: k.qmv4_tc6(x, gu, idss[next(it) % 200], 1, TOPK, G)) + timeit(lambda: k.qmv4_tc6(a, dn, idss[next(it) % 200], TOPK, 1, G, weights=w))
    print(f"T={T} resident: gate_up err {rel(n1, r1):.1e} down err {rel(n2, r2):.1e} | gate_up+down {t_old:6.1f} -> {t_new:6.1f} us")
# streamed: host banks pinned, pool of S slots, plan from offload_plan; compare with qmv4_offload and check installs
hgu, hdn = bank(2 * I, H, E, pin=True), bank(H, I, E, pin=True)
for T in (1, 3):
    res = {}
    for name in ("old", "new"):
        pool_gu = tuple(torch.zeros((S,) + t.shape[1:], dtype=t.dtype, device=dev) for t in hgu)
        pool_dn = tuple(torch.zeros((S,) + t.shape[1:], dtype=t.dtype, device=dev) for t in hdn)
        table = torch.full((E,), -1, dtype=torch.int32, device=dev); owner = torch.full((S,), -1, dtype=torch.int32, device=dev)
        age = torch.full((S,), -1, dtype=torch.long, device=dev); clock = torch.zeros(1, dtype=torch.long, device=dev)
        torch.manual_seed(1); outs = []
        for step in range(30):
            ids = torch.stack([torch.randint(0, 64, (64,), device=dev).unique()[:TOPK] for _ in range(T)]).reshape(-1).int()
            x = torch.randn(T, H, device=dev).bfloat16(); w = torch.rand(T * TOPK, device=dev)
            slots, miss = k.offload_plan(ids, 0, E, table, owner, age, clock)
            hp_gu = tuple(t.data_ptr() for t in hgu); hp_dn = tuple(t.data_ptr() for t in hdn)
            if name == "old":
                g = k.qmv4_offload(x, pool_gu, hp_gu, ids, slots, miss, None, 1, TOPK, G)
                act = torch.nn.functional.silu(g[:, :I]) * g[:, I:]
                o = k.qmv4_offload(act.contiguous(), pool_dn, hp_dn, ids, slots, miss, w, TOPK, 1, G)
            else:
                g = k.qmv4_tc6(x, pool_gu, ids, 1, TOPK, G, host=hp_gu, slots=slots, miss=miss)
                act = torch.nn.functional.silu(g[:, :I]) * g[:, I:]
                o = k.qmv4_tc6(act.contiguous(), pool_dn, ids, TOPK, 1, G, host=hp_dn, slots=slots, miss=miss, weights=w)
            outs.append(o)
        torch.cuda.synchronize()
        res[name] = (torch.cat(outs), pool_gu, pool_dn, owner.clone())
    o_old, pg_old, pd_old, own = res["old"]; o_new, pg_new, pd_new, _ = res["new"]
    same_pool = all(torch.equal(p, q) for p, q in zip(pg_old + pd_old, pg_new + pd_new))
    # pool slots hold exactly the host expert bytes
    ok = all(torch.equal(pg_new[0][s].cpu(), hgu[0][own[s] % E]) for s in range(S) if own[s] >= 0)
    print(f"T={T} streamed: out err vs qmv4_offload {rel(o_new, o_old):.1e}, pools identical {same_pool}, installs match host {ok}")
