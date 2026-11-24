import os, torch
from kernels import get_local_kernel
from pathlib import Path
k = get_local_kernel(Path(os.environ["TRANSFORMERS_MOE_AFFINE_KERNEL"]))
H, I, G, E, L, S, TOPK = 2560, 640, 64, 32, 4, 24, 10
torch.manual_seed(0)
def pinned(*shape, dtype):
    t = torch.empty(*shape, dtype=dtype, pin_memory=True)
    if dtype == torch.int32: t.random_(-2**31, 2**31 - 1)
    else: t.copy_(torch.randn(*shape))
    return t
shapes = [((2 * I, H // 8), torch.int32), ((2 * I, H // G), torch.bfloat16), ((2 * I, H // G), torch.bfloat16),
          ((H, I // 8), torch.int32), ((H, I // G), torch.bfloat16), ((H, I // G), torch.bfloat16)]
host = [[pinned(E, *sh, dtype=dt) for sh, dt in shapes] for _ in range(L)]
pool = [torch.zeros(S, *sh, dtype=dt, device="cuda") for sh, dt in shapes]
handle, mb, res = k.host_tier2_start([t.data_ptr() for layer in host for t in layer], H, I, G, [6, 7], E)
k.host_tier2_installer(handle, [p.data_ptr() for p in pool], [p[0].numel() * p.element_size() for p in pool], E, 0, 8)
table = torch.full((L * E,), -1, dtype=torch.int32, device="cuda"); owner = torch.full((S,), -1, dtype=torch.int32, device="cuda")
age = torch.full((S,), -1, dtype=torch.long, device="cuda"); clock = torch.zeros(1, dtype=torch.long, device="cuda")
rstate = torch.zeros(2, dtype=torch.long, device="cuda")
w = torch.rand(TOPK, device="cuda")
for token in range(60):
    for layer in range(L):
        ids = torch.randperm(E // 2, device="cuda")[:TOPK].int() + (token % 2) * (E // 2) * 0
        slots, gw = k.offload_lookup2(ids, layer, E, table, age, clock, w)
        k.host_refresh2(ids, layer, E, table, owner, age, clock, 4, mb, rstate)
torch.cuda.synchronize()
n = k.host_tier2_drain(handle)
ids = torch.zeros(1, dtype=torch.int32, device="cuda")
k.host_refresh2(ids, 0, E, table, owner, age, clock, 0, mb, rstate)  # commit the rest
torch.cuda.synchronize()
tab, own = table.cpu(), owner.cpu()
committed = bad = stale = pending = 0
for key in range(L * E):
    s = int(tab[key])
    if s == -2: pending += 1
    if s < 0: continue
    committed += 1
    if int(own[s]) != key: stale += 1; continue
    layer, e = divmod(key, E)
    if not all(torch.equal(pool[b][s].cpu(), host[layer][b][e]) for b in range(6)): bad += 1
print(f"installs copied {n}, committed table entries {committed}, pending {pending}, stale {stale}, content mismatches {bad}")
print("OK" if committed > 0 and stale == 0 and bad == 0 and pending == 0 else "FAIL")
k.host_tier2_stop(handle)
