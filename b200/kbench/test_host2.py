"""Host tier v2 vs the GPU kernels on the same banks: 1..4 tokens, 1..40 missed routes."""
import os, torch
from kernels import get_local_kernel
from pathlib import Path
k = get_local_kernel(Path(os.environ["TRANSFORMERS_MOE_AFFINE_KERNEL"]))
H, I, G, E, TOPK = 2560, 640, 64, 32, 10
torch.manual_seed(0)
def pinned(*shape, dtype, scale=0.01):
    t = torch.empty(*shape, dtype=dtype, pin_memory=True)
    if dtype == torch.int32: t.random_(-2**31, 2**31 - 1)
    else: t.copy_(torch.randn(*shape) * scale)
    return t
L = 2
banks, gpu = [], []
for l in range(L):
    b = (pinned(E, 2 * I, H // 8, dtype=torch.int32), pinned(E, 2 * I, H // G, dtype=torch.bfloat16), pinned(E, 2 * I, H // G, dtype=torch.bfloat16),
         pinned(E, H, I // 8, dtype=torch.int32), pinned(E, H, I // G, dtype=torch.bfloat16), pinned(E, H, I // G, dtype=torch.bfloat16))
    banks += [t.data_ptr() for t in b]; gpu.append([t.cuda() for t in b]); gpu[-1].append(b)
handle, mb, res = k.host_tier2_start(banks, H, I, G, list(range(64, 80)), E)
TWO = os.environ.get("TWO") == "1"
handle_b, mb_b, res_b = k.host_tier2_start(banks, H, I, G, list(range(8, 24)), E, 0) if TWO else (None, 0, 0)
state = torch.zeros(3, dtype=torch.long, device="cuda")
table = torch.full((L * E,), -1, dtype=torch.int32, device="cuda")
for ntok in (1, 2, 4):
    for l in range(L):
        ids = torch.stack([torch.randperm(E)[:TOPK] for _ in range(ntok)]).reshape(-1).int().cuda()
        w = torch.rand(ntok * TOPK, device="cuda")
        x = (torch.randn(ntok, H, device="cuda")).bfloat16()
        # make some routes resident (not posted): their weight is applied on the GPU side in the real path
        table.fill_(-1)
        res_experts = torch.randperm(E, device="cuda")[: E // 2]
        table[l * E + res_experts] = 0
        resident = table[l * E + ids.long()] >= 0  # residency is per expert
        gw, gs, gb, dw, ds, db, _ = gpu[l]
        act = torch.nn.functional.silu(k.gather_qmv4(x, gw, gs, gb, ids, TOPK, G)[:, :I].float()).bfloat16().float()
        gu = k.gather_qmv4(x, gw, gs, gb, ids, TOPK, G)
        act = (torch.nn.functional.silu(gu[:, :I].float()).bfloat16().float() * gu[:, I:].float()).bfloat16()
        w_host = w * (~resident)
        ref = k.down_route4(act.contiguous(), dw, ds, db, ids, w_host.float().contiguous(), TOPK, G).float()
        base = torch.randn(ntok, H, device="cuda").bfloat16()
        k.host_post2(ids, w, l, w * (table[l * E + ids.long()] >= 0), x, mb, state, mb_b)
        got = k.host_wait2(base, res, state, res_b).float() - base.float()
        torch.cuda.synchronize()
        err = ((got - ref).norm() / ref.norm()).item()
        print(f"ntok {ntok} layer {l} host routes {int((~resident).sum())}: rel err {err:.2e}")
# nothing missed: state[1] = 0, output = input
table.fill_(0)
base = torch.randn(1, H, device="cuda").bfloat16()
k.host_post2(ids[:TOPK], w[:TOPK], 0, w[:TOPK] * (table[ids[:TOPK].long()] >= 0), x[:1], mb, state)
print("no-miss passthrough:", torch.equal(k.host_wait2(base, res, state, res_b), base))
k.host_tier2_stop(handle)
if TWO: k.host_tier2_stop(handle_b)
