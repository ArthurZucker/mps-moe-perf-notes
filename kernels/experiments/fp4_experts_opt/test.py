"""Correctness of the NVFP4 / MXFP8 expert kernels against an fp32 dequantized reference at Inkling-2-Mini shapes
(H 2304, I 1152, E 128, top-8): T = 1 (decode), 5, 8 and 16 (speculative verify, with experts shared across tokens),
masked routes, every tile of the sweep, and torch.compile through the fakes.

    python ~/Work/hub-kernels/opt/gpu_lock.py python test.py
"""

import torch

import fp4_experts_opt as K


E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
H, I, E, R = 2304, 1152, 128, 8
GATE_CAP, UP_CAP = 4.0, 25.0
dev = "mps"
torch.manual_seed(0)


def dequantize(weight, scale, factor):
    """[E, N, K(/2)] -> fp32 [E, N, K]."""
    if weight.dtype == torch.uint8:
        lut = torch.tensor(E2M1_VALUES, device=weight.device)
        codes = weight.long()
        values = torch.stack((lut[codes & 0xF], lut[codes >> 4]), dim=-1).flatten(-2)
        values = (values.unflatten(-1, (-1, 16)) * scale.float()[..., None]).flatten(-2)
        if factor.dim() == 2:  # gate_up: [gate; up] halves, each with its global scale
            return (values.unflatten(1, (2, -1)) * factor.float()[:, :, None, None]).flatten(1, 2)
        return values * factor.float()[:, None, None]
    return (weight.float().unflatten(-1, (-1, 32)) * torch.exp2(scale.float() - 127)[..., None]).flatten(-2)


def bank(fmt, rows, cols, n=E, gate_up=False):
    if fmt == "nvfp4":
        w = torch.randint(0, 256, (n, rows, cols // 2), dtype=torch.uint8, device=dev)
        s = (torch.rand(n, rows, cols // 16, device=dev) * 0.4 + 0.05).to(torch.float8_e4m3fn)
        f = torch.rand(n, 2 if gate_up else 1, device=dev).squeeze(-1) * 0.1 + 0.02
    else:
        w = (torch.randn(n, rows, cols, device=dev) * 2).to(torch.float8_e4m3fn)
        s = torch.randint(118, 124, (n, rows, cols // 32), dtype=torch.uint8, device=dev)
        f = torch.ones(n, device=dev)  # unused
    return w, s, f


def softcap(v, c):
    return c * torch.tanh(v / c) if c else v


def reference(x, gu, dn, ids, wt, gate_cap=GATE_CAP, up_cap=UP_CAP):
    """x [T, H], ids / wt [T, R] -> out [T, H], act [T * R, I] (act rounded to x's dtype, as the kernels store it)."""
    outs, acts = [], []
    for t in range(x.shape[0]):
        out = torch.einsum("rnk,k->rn", gu[ids[t].long()], x[t].float())
        g, u = out.chunk(2, dim=-1)
        act = softcap(g, gate_cap) * torch.sigmoid(g) * softcap(u, up_cap)
        act = (act * (wt[t] != 0)[:, None]).to(x.dtype).float()
        outs.append((torch.einsum("rhi,ri->rh", dn[ids[t].long()], act) * wt[t, :, None]).sum(0))
        acts.append(act)
    return torch.stack(outs), torch.cat(acts)


def routing(tokens, shared=0.5):
    """[T, R] ids: about `shared` of each token's routes drawn from a common pool (speculative drafts share experts)."""
    pool = torch.randperm(E)[:R]
    rows = []
    for _ in range(tokens):
        keep = pool[torch.randperm(R)[: int(R * shared)]]
        rest = torch.tensor([e for e in torch.randperm(E).tolist() if e not in keep.tolist()][: R - keep.numel()])
        rows.append(torch.cat((keep, rest))[torch.randperm(R)])
    return torch.stack(rows).int().to(dev)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


ok = True


def check(name, err, tol=1e-2):
    global ok
    ok &= err < tol
    print(f"{'ok  ' if err < tol else 'FAIL'} {name:64s} rel err {err:.2e}", flush=True)


for fmt in ("nvfp4", "mxfp8"):
    up, down = bank(fmt, 2 * I, H, gate_up=True), bank(fmt, H, I)
    gu, dn = dequantize(*up), dequantize(*down)
    for dtype in (torch.bfloat16, torch.float16):
        for tokens in (1, 5, 8, 16):
            x = torch.randn(tokens, H, device=dev, dtype=dtype)
            ids = routing(tokens)
            wt = torch.rand(tokens, R, device=dev) + 0.05
            wt[0, 3] = 0.0  # masked routes
            wt[-1, 0] = 0.0
            y_ref, act_ref = reference(x, gu, dn, ids, wt)
            tag = f"{fmt} {str(dtype)[6:]} T={tokens}"
            act = K.gate_up_act(x, *up, ids, wt, GATE_CAP, UP_CAP)
            check(f"gate_up_act {tag}", rel(act, act_ref))
            check(f"down_sum {tag}", rel(K.down_sum(act_ref.to(dtype), *down, ids, wt), y_ref))
            check(f"experts_decode {tag}", rel(K.experts_decode(x, *up, *down, ids, wt, GATE_CAP, UP_CAP), y_ref))
            check(f"experts_fused {tag}", rel(K.experts_fused(x, *up, *down, ids, wt, GATE_CAP, UP_CAP), y_ref))
            # routing tensors as the router produces them: int64 ids, bf16 / fp16 weights (no cast dispatches)
            for wdt in (torch.bfloat16, torch.float16):
                ids64, wtc = ids.long(), wt.to(wdt)
                yc, actc = reference(x, gu, dn, ids, wtc.float())
                check(f"gate_up_act int64 ids {str(wdt)[6:]} weights {tag}",
                      rel(K.gate_up_act(x, *up, ids64, wtc, GATE_CAP, UP_CAP), actc))  # fmt: skip
                check(f"down_sum int64 ids {str(wdt)[6:]} weights {tag}",
                      rel(K.down_sum(actc.to(dtype), *down, ids64, wtc), yc))  # fmt: skip
                check(f"experts_fused int64 ids {str(wdt)[6:]} weights {tag}",
                      rel(K.experts_fused(x, *up, *down, ids64, wtc, GATE_CAP, UP_CAP), yc))  # fmt: skip
        # masked correction pass: 1 of 8 routes active; plain SiLU * up
        x = torch.randn(1, H, device=dev, dtype=dtype)
        ids, wt = routing(1), torch.zeros(1, R, device=dev)
        wt[0, 5] = 0.7
        y_ref, _ = reference(x, gu, dn, ids, wt, 0.0, 0.0)
        check(f"experts_decode masked 1/8 SiLU {fmt} {str(dtype)[6:]}",
              rel(K.experts_decode(x, *up, *down, ids, wt, 0.0, 0.0), y_ref))  # fmt: skip
        check(f"experts_fused masked 1/8 SiLU {fmt} {str(dtype)[6:]}",
              rel(K.experts_fused(x, *up, *down, ids, wt, 0.0, 0.0), y_ref))  # fmt: skip
        errs = [rel(K.experts_fused(x, *up, *down, ids, wt, 0.0, 0.0), y_ref) for _ in range(5)]
        check(f"experts_fused x5 back to back (workspace re-zeroed) {fmt} {str(dtype)[6:]}", max(errs))

    # every tile of the sweep (bfloat16)
    x = torch.randn(8, H, device=dev, dtype=torch.bfloat16)
    ids, wt = routing(8), torch.rand(8, R, device=dev)
    y_ref, act_ref = reference(x, gu, dn, ids, wt)
    for rows in (1, 2):
        for vpl in (8, 16, 32):
            act = K.launch_gate_up(x, *up, ids, wt, GATE_CAP, UP_CAP, tune=True, rows=rows, vpl=vpl)
            check(f"tile gate_up rows={rows} vpl={vpl} {fmt}", rel(act, act_ref))
    for rows, vpl, sg in ((2, 8, 4), (2, 16, 4), (2, 32, 4), (4, 8, 4), (4, 16, 4), (2, 16, 2), (2, 16, 8), (1, 16, 4)):
        out = K.launch_down(act_ref.bfloat16(), *down, ids, wt, tune=True, rows=rows, vpl=vpl, sg=sg)
        check(f"tile down rows={rows} vpl={vpl} sg={sg} {fmt}", rel(out, y_ref))
    for sg, vpl in ((4, 8), (8, 8), (8, 16), (16, 8)):
        out = K.launch_fused(x, *up, *down, ids, wt, GATE_CAP, UP_CAP, tune=True, sg=sg, vpl=vpl)
        check(f"tile fused sg={sg} vpl={vpl} {fmt}", rel(out, y_ref))

    fn = torch.compile(lambda x: K.experts_decode(x, *up, *down, ids[:1], wt[:1], GATE_CAP, UP_CAP), fullgraph=True)
    check(f"torch.compile(experts_decode) == eager {fmt}",
          rel(fn(x[:1]), K.experts_decode(x[:1], *up, *down, ids[:1], wt[:1], GATE_CAP, UP_CAP)), 1e-6)  # fmt: skip

print("ALL OK" if ok else "SOME FAILED")
raise SystemExit(0 if ok else 1)
