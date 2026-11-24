"""Correctness vs the Triton gather_qmv reference + bandwidth, at Qwen3.8 shapes (512 experts, K/N 2560/1280, 640/2560)."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_qmv4  # noqa
from transformers.integrations.triton_affine_moe import gather_qmv
torch.manual_seed(0)
E, H, I, topk, G = 512, 2560, 640, 10, 64
def bank(N, K):
    w = torch.randint(-2**31, 2**31 - 1, (E, N, K // 8), dtype=torch.int32, device="cuda")
    s = (torch.rand(E, N, K // G, device="cuda") * 0.02).bfloat16()
    b = (torch.randn(E, N, K // G, device="cuda") * 0.05).bfloat16()
    return w, s, b
gu, dn = bank(2 * I, H), bank(H, I)
def ev(f, it=50):
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20): f()
    g.replay(); torch.cuda.synchronize(); st.record()
    for _ in range(it): g.replay()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / it / 20 * 1e3
_old = lambda f, it=200: (lambda st, en: ([f() for _ in range(5)], torch.cuda.synchronize(), st.record(), [f() for _ in range(it)], en.record(), torch.cuda.synchronize(), st.elapsed_time(en) / it * 1e3)[-1])(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
for T in (1, 4):
    x = torch.randn(T, H, device="cuda").bfloat16()
    ids = torch.stack([torch.randperm(E, device="cuda")[:topk] for _ in range(T)]).reshape(-1).int()
    wts = torch.rand(T * topk, device="cuda")
    ref = gather_qmv(x.repeat_interleave(topk, 0), *gu, ids, G, 4).float()
    got = torch.ops.moe_cuda.gather_qmv4(x, *gu, ids, topk, G).float()
    print(f"T={T} gate_up rel err {((got - ref).norm() / ref.norm()).item():.2e}")
    act = (torch.randn(T * topk, I, device="cuda") * 0.5).bfloat16()
    refd = (gather_qmv(act, *dn, ids, G, 4).float() * wts[:, None]).view(T, topk, H).sum(1)
    gotd = torch.ops.moe_cuda.down_route4(act, *dn, ids, wts, topk, G).float()
    print(f"T={T} down_route rel err {((gotd - refd).norm() / refd.norm()).item():.2e}")
    by_gu = T * topk * gu[0][0].numel() * 4 * (1 + 2 / 16)  # codes + scale/bias (2 bf16 per 64 codes = 4B per 32B)
    by_dn = T * topk * dn[0][0].numel() * 4 * (1 + 2 / 16)
    for name, f, by in (("triton gate_up", lambda: gather_qmv(x.repeat_interleave(topk, 0), *gu, ids, G, 4), by_gu),
                        ("cuda gate_up", lambda: torch.ops.moe_cuda.gather_qmv4(x, *gu, ids, topk, G), by_gu),
                        ("cuda down_route", lambda: torch.ops.moe_cuda.down_route4(act, *dn, ids, wts, topk, G), by_dn)):
        us = ev(f)
        print(f"  T={T} {name:16s} {us:8.1f} us  {by / us / 1e3:7.0f} GB/s")
