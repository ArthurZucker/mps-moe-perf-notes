"""NVFP4 / MXFP8 expert kernels at Inkling-2-Mini shapes (H 2304, I 1152, E 128, top-8): this module vs the current
~/Work/hub-kernels/opt/fp4/fp4_experts.py, through the shared GPU lock.

  isolated   each kernel alone, cycling distinct expert banks so reads come from DRAM (best of 5 trials);
             % of a 250 GB/s roofline over the bytes of the distinct experts read
  sequence   a realistic decode step: per layer an RMSNorm, the router matmul + top-k + softmax, the experts and the
             residual add, cycling banks in the model's 11 NVFP4 : 1 MXFP8 ratio; the experts' cost in context is the
             step minus the same step without them

    python ~/Work/hub-kernels/opt/gpu_lock.py python bench.py [--tokens 1 8] [--sweep] [--peak 250]
"""

import argparse
import pathlib
import sys
import time

import torch

import fp4_experts_opt as K


sys.path.insert(0, str(pathlib.Path.home() / "Work/hub-kernels/opt/fp4"))
import fp4_experts as old  # noqa: E402


p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 16])
p.add_argument("--peak", type=float, default=250.0)
p.add_argument("--sweep", action="store_true")
p.add_argument("--no-sequence", action="store_true")
p.add_argument("--sequence-only", action="store_true")
args = p.parse_args()

H, I, E, R = 2304, 1152, 128, 8
GATE_CAP, UP_CAP = 4.0, 25.0
dev, bf16 = "mps", torch.bfloat16
torch.manual_seed(0)
BANKS = {"nvfp4": 6, "mxfp8": 3}


def bank(fmt, rows, cols, gate_up=False):
    if fmt == "nvfp4":
        w = torch.randint(0, 256, (E, rows, cols // 2), dtype=torch.uint8, device=dev)
        s = (torch.rand(E, rows, cols // 16, device=dev) * 0.4 + 0.05).to(torch.float8_e4m3fn)
        f = torch.rand(E, 2 if gate_up else 1, device=dev).squeeze(-1) * 0.1 + 0.02
    else:
        w = torch.randint(0, 256, (E, rows, cols), dtype=torch.uint8, device=dev)
        w = torch.where((w & 0x7F) == 0x7F, w - 1, w).view(torch.float8_e4m3fn)  # no NaN codes
        s = torch.randint(118, 124, (E, rows, cols // 32), dtype=torch.uint8, device=dev)
        f = torch.ones(E, device=dev)
    return w, s, f


def expert_bytes(fmt):
    """(gate_up, down) bytes of one expert."""
    per = (lambda n, k: n * (k // 2 + k // 16)) if fmt == "nvfp4" else (lambda n, k: n * (k + k // 32))
    return per(2 * I, H), per(H, I)


def routing(tokens, shared):
    pool = torch.randperm(E)[:R]
    rows = []
    for _ in range(tokens):
        keep = pool[torch.randperm(R)[: int(R * shared)]]
        rest = [e for e in torch.randperm(E).tolist() if e not in keep.tolist()][: R - keep.numel()]
        rows.append(torch.cat((keep, torch.tensor(rest)))[torch.randperm(R)])
    return torch.stack(rows).int().to(dev)


def timed(fns, reps=10, trials=5):
    for f in fns:
        f()
    torch.mps.synchronize()
    best = float("inf")
    for _ in range(trials):
        t = time.perf_counter()
        for _ in range(reps):
            for f in fns:
                f()
        torch.mps.synchronize()
        best = min(best, (time.perf_counter() - t) / reps / len(fns) * 1e6)
    return best


def line(label, us, nbytes=None, extra=""):
    bw = f"{nbytes / us / 1e3:6.0f} GB/s {nbytes / us / 1e3 / args.peak * 100:5.1f}%" if nbytes else " " * 19
    print(f"  {label:66s} {us:8.1f} us {bw}{extra}", flush=True)


banks = {fmt: [(bank(fmt, 2 * I, H, True), bank(fmt, H, I)) for _ in range(n)] for fmt, n in BANKS.items()}
start = time.perf_counter()

# ---- isolated ----
for tokens in [] if args.sequence_only else args.tokens:
    for shared in (0.0, 0.5) if tokens > 1 else (0.0,):
        x = torch.randn(tokens, H, device=dev, dtype=bf16)
        ids = routing(tokens, shared)
        wt = torch.rand(tokens, R, device=dev) + 0.05
        distinct = ids.unique().numel()
        print(f"T={tokens} R={R}, {distinct} distinct experts ({tokens * R} pairs), elapsed "
              f"{time.perf_counter() - start:.0f}s", flush=True)  # fmt: skip
        act = old.gate_up_act(x, *banks["nvfp4"][0][0], ids, wt, GATE_CAP, UP_CAP)
        for fmt in ("nvfp4", "mxfp8"):
            gb, db = expert_bytes(fmt)
            bs = banks[fmt]
            run = lambda fn: timed([lambda b=b: fn(b) for b in bs])  # noqa: E731
            cases = [
                ("old gate_up_act", lambda b: old.gate_up_act(x, *b[0], ids, wt, GATE_CAP, UP_CAP), distinct * gb),
                ("new gate_up_act", lambda b: K.gate_up_act(x, *b[0], ids, wt, GATE_CAP, UP_CAP), distinct * gb),
                ("old down_sum", lambda b: old.down_sum(act, *b[1], ids, wt), distinct * db),
                ("new down_sum", lambda b: K.down_sum(act, *b[1], ids, wt), distinct * db),
                ("old both", lambda b: old.down_sum(old.gate_up_act(x, *b[0], ids, wt, GATE_CAP, UP_CAP), *b[1], ids, wt),
                 distinct * (gb + db)),
                ("new experts_decode (2 dispatches)",
                 lambda b: K.experts_decode(x, *b[0], *b[1], ids, wt, GATE_CAP, UP_CAP), distinct * (gb + db)),
                ("new experts_fused (1 dispatch)",
                 lambda b: K.experts_fused(x, *b[0], *b[1], ids, wt, GATE_CAP, UP_CAP), distinct * (gb + db)),
            ]  # fmt: skip
            if tokens == 1:
                wm = torch.zeros_like(wt)
                wm[0, 2] = 1.0
                cases += [
                    ("old both, masked 1/8",
                     lambda b: old.down_sum(old.gate_up_act(x, *b[0], ids, wm, GATE_CAP, UP_CAP), *b[1], ids, wm), gb + db),
                    ("new experts_decode, masked 1/8",
                     lambda b: K.experts_decode(x, *b[0], *b[1], ids, wm, GATE_CAP, UP_CAP), gb + db),
                    ("new experts_fused, masked 1/8",
                     lambda b: K.experts_fused(x, *b[0], *b[1], ids, wm, GATE_CAP, UP_CAP), gb + db),
                ]  # fmt: skip
            if args.sweep:
                for rows in (1, 2):
                    for vpl in (8, 16, 32):
                        cases.append((f"sweep gate_up rows={rows} vpl={vpl}",
                                      lambda b, t=dict(rows=rows, vpl=vpl): K.launch_gate_up(
                                          x, *b[0], ids, wt, GATE_CAP, UP_CAP, tune=True, **t),
                                      distinct * gb))  # fmt: skip
                for sg, vpl in ((4, 8), (8, 8), (8, 16), (16, 8)):
                    cases.append((f"sweep fused sg={sg} vpl={vpl}",
                                  lambda b, t=dict(sg=sg, vpl=vpl): K.launch_fused(x, *b[0], *b[1], ids, wt, GATE_CAP,
                                                                                   UP_CAP, tune=True, **t),
                                  distinct * (gb + db)))  # fmt: skip
                for rows, vpl, sg in ((2, 8, 4), (2, 16, 4), (2, 32, 4), (4, 8, 4), (4, 16, 4), (2, 16, 2), (2, 16, 8),
                                      (1, 16, 4)):  # fmt: skip
                    cases.append((f"sweep down rows={rows} vpl={vpl} sg={sg}",
                                  lambda b, t=dict(rows=rows, vpl=vpl, sg=sg): K.launch_down(act, *b[1], ids, wt,
                                                                                             tune=True, **t),
                                  distinct * db))  # fmt: skip
            for i, (label, fn, nbytes) in enumerate(cases):
                line(f"[{i + 1}/{len(cases)}] {fmt} T={tokens} shared={shared}: {label}", run(fn), nbytes)

# ---- realistic decode-step sequence ----
if not args.no_sequence:
    layers = 48
    order = ["mxfp8" if i % 12 == 11 else "nvfp4" for i in range(layers)]  # 44 NVFP4 + 4 MXFP8
    router = [torch.randn(H, E, device=dev, dtype=bf16) * 0.02 for _ in range(4)]
    norm_w = torch.ones(H, device=dev, dtype=bf16)
    for tokens in args.tokens:
        x0 = torch.randn(tokens, H, device=dev, dtype=bf16)

        def step(experts):
            h = x0
            for li, fmt in enumerate(order):
                b = banks[fmt][li % BANKS[fmt]]
                n = h.float()
                n = (n * torch.rsqrt(n.pow(2).mean(-1, keepdim=True) + 1e-6)).to(bf16) * norm_w
                logits = n @ router[li % 4]
                wt, ids = torch.topk(logits.float().softmax(-1), R, dim=-1)
                wt = wt.to(bf16)  # as the model hands them over: int64 ids, bf16 weights
                h = h + experts(n, b, ids, wt) if experts else h + n
            return h

        def casts_two(gu, dn):
            def fn(n, b, ids, wt):
                ids, wt = ids.int(), wt.float()  # the caller's casts: two dispatches
                return dn(gu(n, *b[0], ids, wt, GATE_CAP, UP_CAP), *b[1], ids, wt)
            return fn

        cases = {
            "router/norm/residual only": None,
            "old: casts + gate_up_act + down_sum": casts_two(old.gate_up_act, old.down_sum),
            "new: casts + gate_up_act + down_sum": casts_two(K.gate_up_act, K.down_sum),
            "new: gate_up_act + down_sum, no casts": lambda n, b, ids, wt: K.down_sum(
                K.gate_up_act(n, *b[0], ids, wt, GATE_CAP, UP_CAP), *b[1], ids, wt),
            "new: experts_fused (1 dispatch, no casts)": lambda n, b, ids, wt: K.experts_fused(
                n, *b[0], *b[1], ids, wt, GATE_CAP, UP_CAP),
        }  # fmt: skip
        times = {}
        for i, (label, fn) in enumerate(cases.items()):
            times[label] = timed([lambda fn=fn: step(fn)], reps=5, trials=7) / 1e3
            cost = times[label] - times["router/norm/residual only"]
            ctx = f"  experts in context {cost:6.2f} ms = {cost / layers * 1e3:6.1f} us/eval" if fn else ""
            print(f"  [{i + 1}/{len(cases)}] sequence T={tokens}, {layers} layers: {label:42s} "
                  f"{times[label]:7.2f} ms/step{ctx}", flush=True)  # fmt: skip
print(f"done in {time.perf_counter() - start:.0f}s", flush=True)
