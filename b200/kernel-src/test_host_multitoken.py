import torch, sys
from pathlib import Path
from kernels import get_local_kernel
k = get_local_kernel(Path(sys.argv[1]))
torch.manual_seed(0)
E, H, I, G, topk = 64, 2560, 640, 64, 10
def bank(N, Kd):
    w = torch.randint(-2**31, 2**31 - 1, (E, N, Kd // 8), dtype=torch.int32)
    s = (torch.rand(E, N, Kd // G) * 0.02).bfloat16(); b = (torch.randn(E, N, Kd // G) * 0.05).bfloat16()
    return [t.pin_memory() for t in (w, s, b)]
hg, hd = bank(2 * I, H), bank(H, I)
dg, dd = [t.cuda() for t in hg], [t.cuda() for t in hd]
handle, mb, res = k.host_tier_start([t.data_ptr() for t in (*hg, *hd)], H, I, G, list(range(8, 32)))
state = torch.zeros(2, dtype=torch.long, device="cuda")
table = torch.full((E,), -1, dtype=torch.int32, device="cuda"); table[:32] = torch.arange(32, dtype=torch.int32, device="cuda")
age = torch.full((E,), -1, dtype=torch.long, device="cuda"); clock = torch.zeros(1, dtype=torch.long, device="cuda")
worst = 0
for step in range(24):
    T = 1 + step % 4
    ids = torch.stack([torch.randperm(E, device="cuda")[:topk] for _ in range(T)]).reshape(-1).int()
    w = torch.rand(T * topk, device="cuda"); x = torch.randn(T, H, device="cuda").bfloat16()
    slots, gw = k.offload_lookup(ids, 0, E, table, age, clock, w)
    k.host_post(ids, w, 0, E, table, x, mb, state)
    gu = k.qmv4_v2(x, dg, slots, 1, topk, G, rows=2)
    act = (torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]).contiguous()
    out = k.host_wait(k.qmv4_v2(act, dd, slots, topk, 1, G, weights=gw, rows=1), res, state)
    ref_gu = k.gather_qmv4(x, *dg, ids, topk, G)
    ref = k.down_route4((torch.nn.functional.silu(ref_gu[:, :I]) * ref_gu[:, I:]).contiguous(), *dd, ids, w, topk, G)
    worst = max(worst, ((out.float() - ref.float()).norm() / ref.norm()).item())
print(f"multi-token host tier vs resident (T=1..4, ~half the experts on the host): max rel err {worst:.2e}")
k.host_tier_stop(handle)
