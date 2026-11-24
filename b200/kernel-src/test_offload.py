import torch, sys, time
from pathlib import Path
from kernels import get_local_kernel
k = get_local_kernel(Path(sys.argv[1]))
torch.manual_seed(0)
L, E, H, I, topk, G, S = 3, 64, 2560, 640, 10, 64, 40
def bank(N, K, dev):
    w = torch.randint(-2**31, 2**31 - 1, (E, N, K // 8), dtype=torch.int32)
    s = (torch.rand(E, N, K // G) * 0.02).bfloat16(); b = (torch.randn(E, N, K // G) * 0.05).bfloat16()
    return [t.to(dev) if dev == "cuda" else t.pin_memory() for t in (w, s, b)]
host = [(bank(2 * I, H, "cpu"), bank(H, I, "cpu")) for _ in range(L)]
dev = [tuple(tuple(t.cuda() for t in b) for b in hb) for hb in host]
pool_gu = [torch.zeros(S, 2 * I, H // 8, dtype=torch.int32, device="cuda"), torch.zeros(S, 2 * I, H // G, dtype=torch.bfloat16, device="cuda"), torch.zeros(S, 2 * I, H // G, dtype=torch.bfloat16, device="cuda")]
pool_dn = [torch.zeros(S, H, I // 8, dtype=torch.int32, device="cuda"), torch.zeros(S, H, I // G, dtype=torch.bfloat16, device="cuda"), torch.zeros(S, H, I // G, dtype=torch.bfloat16, device="cuda")]
table = torch.full((L * E,), -1, dtype=torch.int32, device="cuda"); owner = torch.full((S,), -1, dtype=torch.int32, device="cuda")
age = torch.full((S,), -1, dtype=torch.long, device="cuda"); clock = torch.zeros(1, dtype=torch.long, device="cuda")
maxerr = 0; misses = 0; calls = 0
for step in range(60):
    for l in range(L):
        x = torch.randn(1, H, device="cuda").bfloat16()
        ids = torch.randperm(E, device="cuda")[:topk].int() if step % 3 else torch.arange(topk, device="cuda").int()  # some reuse
        w = torch.rand(topk, device="cuda")
        slots, miss = k.offload_plan(ids, l, E, table, owner, age, clock)
        gu = k.qmv4_offload(x, pool_gu, [t.data_ptr() for t in host[l][0]], ids, slots, miss, None, 1, topk, G)
        act = (gu[:, :I].float().sigmoid() * gu[:, I:].float()).bfloat16().contiguous()
        out = k.qmv4_offload(act, pool_dn, [t.data_ptr() for t in host[l][1]], ids, slots, miss, w, topk, 1, G)
        ref_gu = k.gather_qmv4(x, *dev[l][0], ids, topk, G)
        ref = k.down_route4(act, *dev[l][1], ids, w, topk, G)
        maxerr = max(maxerr, (gu.float() - ref_gu.float()).abs().max().item(), (out.float() - ref.float()).abs().max().item())
        misses += miss.sum().item(); calls += 1
# consistency: every resident slot holds its expert's data
tb = table.view(L, E).cpu(); bad = 0
for l in range(L):
    for e in range(E):
        s = tb[l, e].item()
        if s >= 0:
            bad += not torch.equal(pool_gu[0][s].cpu(), host[l][0][0][e]) or not torch.equal(pool_dn[1][s].cpu(), host[l][1][1][e])
print(f"max abs diff vs resident kernels {maxerr}  misses/call {misses/calls:.2f}  resident {int((tb>=0).sum())}/{S}  bad slots {bad}")
# PCIe throughput of an all-miss call (fresh pool), real expert size
for name, R in (("1 route", 1), ("10 routes", 10)):
    table.fill_(-1); owner.fill_(-1); age.fill_(-1)
    ids = torch.arange(R, device="cuda").int(); x = torch.randn(1, H, device="cuda").bfloat16()
    torch.cuda.synchronize(); t = time.perf_counter()
    slots, miss = k.offload_plan(ids, 0, E, table, owner, age, clock)
    gu = k.qmv4_offload(x, pool_gu, [t.data_ptr() for t in host[0][0]], ids, slots, miss, None, 1, R, G)
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    by = R * (pool_gu[0][0].numel() * 4 + pool_gu[1][0].numel() * 4)
    print(f"all-miss gate_up {name}: {dt*1e6:.0f} us, {by/dt/1e9:.1f} GB/s over PCIe")
