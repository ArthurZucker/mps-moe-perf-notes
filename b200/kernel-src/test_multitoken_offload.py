"""Offload plan + matvecs for T tokens with duplicated experts across tokens == the resident kernels."""
import torch, sys
from pathlib import Path
from kernels import get_local_kernel
k = get_local_kernel(Path(sys.argv[1]))
torch.manual_seed(0)
E, H, I, G, topk, S = 64, 2560, 640, 64, 10, 24
def bank(N, Kd):
    w = torch.randint(-2**31, 2**31 - 1, (E, N, Kd // 8), dtype=torch.int32)
    s = (torch.rand(E, N, Kd // G) * 0.02).bfloat16(); b = (torch.randn(E, N, Kd // G) * 0.05).bfloat16()
    return [t.pin_memory() for t in (w, s, b)]
hg, hd = bank(2 * I, H), bank(H, I)
dg, dd = [t.cuda() for t in hg], [t.cuda() for t in hd]
pg = [torch.zeros(S, *t.shape[1:], dtype=t.dtype, device="cuda") for t in hg]
pd = [torch.zeros(S, *t.shape[1:], dtype=t.dtype, device="cuda") for t in hd]
table = torch.full((E,), -1, dtype=torch.int32, device="cuda"); owner = torch.full((S,), -1, dtype=torch.int32, device="cuda")
age = torch.full((S,), -1, dtype=torch.long, device="cuda"); clock = torch.zeros(1, dtype=torch.long, device="cuda")
ok = True
for step in range(30):
    T = 1 + step % 4
    ids = torch.stack([torch.randperm(16, device="cuda")[:topk] for _ in range(T)]).reshape(-1).int()  # 16 experts: heavy overlap
    w = torch.rand(T * topk, device="cuda"); x = torch.randn(T, H, device="cuda").bfloat16()
    slots, miss = k.offload_plan(ids, 0, E, table, owner, age, clock)
    gu = k.qmv4_offload(x, pg, [t.data_ptr() for t in hg], ids, slots, miss, None, 1, topk, G)
    act = (torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]).contiguous()
    out = k.qmv4_offload(act, pd, [t.data_ptr() for t in hd], ids, slots, miss, w, topk, 1, G)
    ref_gu = k.gather_qmv4(x, *dg, ids, topk, G)
    ref = k.down_route4(act, *dd, ids, w, topk, G)
    ok &= torch.equal(gu, ref_gu) and torch.equal(out, ref)
tb = table.cpu(); bad = sum(1 for e in range(E) if tb[e] >= 0 and not torch.equal(pg[0][tb[e]].cpu(), hg[0][e]))
print(f"multi-token offload == resident over 30 calls (T=1..4, overlapping experts): {ok}; corrupt slots: {bad}; resident {int((tb >= 0).sum())}/{S}")
