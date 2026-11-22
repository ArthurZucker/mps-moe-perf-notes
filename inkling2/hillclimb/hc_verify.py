"""Correctness of every custom Metal kernel the accepted configs use, against a torch reference at Inkling-2-Mini's real
shapes (pinned hub revision). One line per check: max relative error and the tolerance it must stay under.

    PYTHONPATH=src:benchmark_v2/inkling2_metal:benchmark_v2/inkling2_metal/hillclimb python hc_verify.py
"""
import sys
from hc_bench import gpu_busy, on_ac
if not on_ac() or gpu_busy() > 15:
    sys.exit(f"refusing to run: battery or busy GPU ({gpu_busy()}%)")
import torch
from transformers.integrations.hub_kernels import metal_kernel

K = metal_kernel("metal-moe-decode")
mods = sys.modules
torch.manual_seed(0)
H, I, E, R = 2304, 1152, 128, 8
dev, bf = "mps", torch.bfloat16
results = []


def check(name, ref, out, tol):
    ref, out = ref.float(), out.float()
    err = ((ref - out).abs().max() / ref.abs().max().clamp_min(1e-12)).item()
    ok = err <= tol and torch.isfinite(out).all().item()
    results.append(ok)
    print(f"[{len(results)}] {'ok  ' if ok else 'FAIL'} {name:60s} rel err {err:.2e} (tol {tol:.0e})", flush=True)


def deq_nvfp4(w, s, f):  # reference dequantization: E2M1 codes, e4m3 block scales, fp32 global
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    lo, hi = lut[(w & 0xF).long()], lut[(w >> 4).long()]
    vals = torch.stack([lo, hi], -1).flatten(-2)
    return vals * s.float().repeat_interleave(16, -1) * f


# 1. router v2 vs torch
for trial in range(4):
    x = torch.randn(1, H, device=dev, dtype=bf); w = torch.randn(E, H, device=dev) * 0.02; b = torch.randn(E, device=dev) * 0.01
    lo, wt, ix = K.router_topk(x, w, b, R, 2.5)
    rl = x.float() @ w.t(); sc = torch.sigmoid(rl); rix = torch.topk(sc + b, R).indices
    rw = sc.gather(1, rix); rw = rw / rw.sum(-1, keepdim=True) * 2.5
    same = set(ix[0].tolist()) == set(rix[0].tolist())
    results.append(same); print(f"[{len(results)}] {'ok  ' if same else 'FAIL'} router_topk indices (trial {trial})", flush=True)
    check(f"router_topk logits (trial {trial})", rl, lo, 1e-5)

# 2. NVFP4 experts decode (gate_up_act + down_route + fixed-order sum) vs dequantized torch
gw = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
gs = (torch.rand(E, 2 * I, H // 16, device=dev) * 0.4 + 0.05).to(torch.float8_e4m3fn)
gf = torch.rand(E, 2, device=dev) * 0.05 + 0.01
dw = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
ds = (torch.rand(E, H, I // 16, device=dev) * 0.4 + 0.05).to(torch.float8_e4m3fn)
df = torch.rand(E, device=dev) * 0.05 + 0.01
for T in (1, 3):
    x = torch.randn(T, H, device=dev, dtype=bf) * 0.5
    wts, ids = torch.topk(torch.rand(T, E, device=dev), R, dim=-1)
    act = K.fp4_gate_up_act(x, gw, gs, gf, ids, wts, 4.0, 25.0)
    y = K.fp4_down_route(act, dw, ds, df, ids, wts).view(T, R, H).sum(1)
    ref = torch.zeros(T, H, device=dev)
    for t in range(T):
        for r in range(R):
            e = ids[t, r].item()
            g = deq_nvfp4(gw[e, :I], gs[e, :I], gf[e, 0]) @ x[t].float()
            u = deq_nvfp4(gw[e, I:], gs[e, I:], gf[e, 1]) @ x[t].float()
            a = (4 * torch.tanh(g / 4)) * torch.sigmoid(g) * (25 * torch.tanh(u / 25))
            ref[t] += wts[t, r] * (deq_nvfp4(dw[e], ds[e], df[e]) @ a.to(bf).float())
    check(f"nvfp4 experts decode (gate_up_act + down_route), T={T}", ref, y, 2e-2)
    yp = K.fp4_experts_prefill(x, gw, gs, gf, dw, ds, df, ids, wts, 4.0, 25.0) if T >= 1 else None
    check(f"nvfp4 experts prefill GEMM vs reference, T={T}", ref, yp, 2e-2)

# 3. qmv4_multi vs three qmv4
def q4(n):
    return (torch.randint(0, 2**31, (n, H // 8), dtype=torch.int32, device=dev).view(torch.uint32),
            torch.rand(n, H // 64, device=dev, dtype=bf) * 0.01, torch.rand(n, H // 64, device=dev, dtype=bf) * 0.01 - 0.005)
qs = [q4(6552), q4(1024), q4(1024)]; xs = [torch.randn(1, H, device=dev, dtype=bf) for _ in range(3)]
outs = mods[K.__name__ + ".qmv4"].qmv4_multi(xs[0], *qs[0], xs[1], *qs[1], xs[2], *qs[2], 64)
for i, (x, q, o) in enumerate(zip(xs, qs, outs)):
    check(f"qmv4_multi output {i} vs qmv4", K.qmv4(x, *q, 64), o, 0.0)

# 4. prefill attention vs the eager reference (bias, window, sinks)
HQ, HKV, D, RE = 24, 8, 128, 1024
for T, window in ((512, 1024), (777, None), (1500, 1024)):
    L = -(-T // 32) * 32
    q = torch.randn(1, HQ, T, D, device=dev, dtype=bf); k = torch.randn(1, HKV, L, D, device=dev, dtype=bf); v = torch.randn_like(k)
    rs = torch.randn(1, T, HQ, 16, device=dev) * 0.5; proj = torch.randn(16, RE, device=dev) * 0.3; sk = torch.randn(1, T, HQ, device=dev)
    dist = torch.arange(T, device=dev)[:, None] - torch.arange(L, device=dev)[None, :]
    rel = (rs @ proj).transpose(1, 2); bias = rel.gather(-1, dist.clamp(0, RE - 1).expand(1, HQ, -1, -1)).masked_fill((dist < 0) | (dist >= RE), 0)
    valid = (dist >= 0) & ((dist < window) if window else True)
    wgt = (q @ k.repeat_interleave(3, 1).transpose(2, 3)) * D ** -0.5 + bias.to(bf)
    wgt = wgt.masked_fill(~valid, torch.finfo(bf).min)
    p = torch.softmax(torch.cat([wgt, sk.transpose(1, 2).unsqueeze(-1).to(bf)], -1).float(), -1)[..., :-1].to(bf)
    ref = (p @ v.repeat_interleave(3, 1)).transpose(1, 2)
    out = mods[K.__name__ + ".prefill_attention"].prefill_attention(q, k, v, rs, proj, dist, sk, D ** -0.5, window)
    check(f"prefill attention T={T} window={window}", ref, out, 1.5e-2)

print(f"\n{sum(results)}/{len(results)} checks passed", flush=True)
sys.exit(0 if all(results) else 1)
