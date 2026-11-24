import os, torch
from kernels import get_local_kernel
from pathlib import Path
k = get_local_kernel(Path(os.environ["TRANSFORMERS_MOE_AFFINE_KERNEL"]))
H, I, G, E, L, S = 2560, 640, 64, 32, 4, 24
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
table = torch.full((L * E,), -1, dtype=torch.int32, device="cuda"); owner = torch.full((S,), -1, dtype=torch.int32, device="cuda")
age = torch.full((S,), -1, dtype=torch.long, device="cuda"); clock = torch.zeros(1, dtype=torch.long, device="cuda")
rstate = torch.zeros(2, dtype=torch.long, device="cuda"); hstate = torch.zeros(1, dtype=torch.long, device="cuda")
k.host_tier2_installer(handle, [p.data_ptr() for p in pool], [p[0].numel() * p.element_size() for p in pool], E, 0, 8, table.data_ptr())
scratch = torch.zeros(E, device="cuda"); counter = torch.zeros(1, dtype=torch.int32, device="cuda")
W = torch.randn(E, H, device="cuda").bfloat16(); dummy = torch.zeros(1, dtype=torch.int32, device="cuda")
w = torch.rand(10, device="cuda")
hits_after_prefetch = guessed = 0
for token in range(40):
    for layer in range(L):
        x = torch.randn(1, H, device="cuda").bfloat16()
        if layer > 0:
            k.host_guess2(x, W, 6, layer, table, mb, hstate, scratch, counter, 2, 8)
            k.host_refresh2(dummy, layer, E, table, owner, age, clock, 2, mb, rstate, 1)
        torch.cuda.synchronize(); import time; time.sleep(0.002)  # the copy engine lands the prefetch (stand-in for attention)
        ids = (W.float() @ x.float().T).squeeze(1).topk(10).indices.int()
        if layer > 0:
            top2 = ids[:2]
            guessed += 2; hits_after_prefetch += int((table[layer * E + top2.long()] >= 0).sum())
        slots, gw = k.offload_lookup2(ids, layer, E, table, age, clock, w)
        k.host_refresh2(ids, layer, E, table, owner, age, clock, 2, mb, rstate, 0)
torch.cuda.synchronize(); k.host_tier2_drain(handle)
k.host_refresh2(dummy, 0, E, table, owner, age, clock, 0, mb, rstate, 0); torch.cuda.synchronize()
tab, own = table.cpu(), owner.cpu()
bad = stale = pending = committed = 0
for key in range(L * E):
    s = int(tab[key])
    if s == -2: pending += 1
    if s < 0: continue
    committed += 1
    if int(own[s]) != key: stale += 1; continue
    l, e = divmod(key, E)
    if not all(torch.equal(pool[b][s].cpu(), host[l][b][e]) for b in range(6)): bad += 1
print(f"prefetched top-2 already resident at lookup: {hits_after_prefetch}/{guessed}; committed {committed}, pending {pending}, stale {stale}, mismatches {bad}")
print("OK" if stale == 0 and bad == 0 and pending == 0 and hits_after_prefetch > 0 else "FAIL")
k.host_tier2_stop(handle)
