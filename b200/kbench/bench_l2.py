"""Expert matvec microbench at the real Qwen3.8 shapes (H=2560, I=640, 512 experts, top-10, g64): achieved HBM GB/s of
gather_qmv4 (gate_up) and down_route4 for T tokens, ids redrawn per call so weights come from HBM, not L2."""
import os, sys, torch
from kernels import get_local_kernel
from pathlib import Path
k = get_local_kernel(Path(os.environ["TRANSFORMERS_MOE_AFFINE_KERNEL"]))
E, H, I, TOPK, G = 512, 2560, 640, 10, 64
dev = "cuda"
torch.manual_seed(0)
wgu = torch.randint(-2**31, 2**31 - 1, (E, 2 * I, H // 8), dtype=torch.int32, device=dev)
sgu = (torch.randn(E, 2 * I, H // G, device=dev) * 0.01).bfloat16(); bgu = (torch.randn(E, 2 * I, H // G, device=dev) * 0.01).bfloat16()
wd = torch.randint(-2**31, 2**31 - 1, (E, H, I // 8), dtype=torch.int32, device=dev)
sd = (torch.randn(E, H, I // G, device=dev) * 0.01).bfloat16(); bd = (torch.randn(E, H, I // G, device=dev) * 0.01).bfloat16()
B_EXP = 2 * I * H // 2 + 2 * (2 * I * H // G) * 2  # gate_up bytes per expert
B_DN = H * I // 2 + 2 * (H * I // G) * 2
fns = {"gate_up": lambda x, ids, w: k.gather_qmv4(x, wgu, sgu, bgu, ids, TOPK, G),
       "down": lambda a, ids, w: k.down_route4(a, wd, sd, bd, ids, w, TOPK, G)}
for R in (1, 2, 4, 8):
    fns[f"gu_mma{R}"] = (lambda R: lambda x, ids, w: k.gather_qmv4_mma(x, wgu, sgu, bgu, ids, TOPK, G, R))(R)
    fns[f"dn_mma{R}"] = (lambda R: lambda a, ids, w: k.down_route4_mma(a, wd, sd, bd, ids, w, TOPK, G, R))(R)
for R in ():
    fns[f"gu_v3r{R}"] = (lambda R: lambda x, ids, w: k.gather_qmv4_v3(x, wgu, sgu, bgu, ids, TOPK, G, R))(R)
    fns[f"dn_v3r{R}"] = (lambda R: lambda a, ids, w: k.down_route4_v3(a, wd, sd, bd, ids, w, TOPK, G, R))(R)
extra = {}
if len(sys.argv) > 1:
    extra = dict(a.split("=") for a in sys.argv[1:])
for T in (1, 2, 3, 4):
    N_IT = 400
    idss0 = [torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)]).reshape(-1).int() for _ in range(N_IT)]; idss = [idss0[0]] * N_IT if os.environ.get("L2") else idss0
    uniq = sum(len(torch.unique(i)) for i in idss) / N_IT
    x = torch.randn(T, H, device=dev).bfloat16(); a = torch.randn(T * TOPK, I, device=dev).bfloat16()
    w = torch.rand(T * TOPK, device=dev)
    for name, f in fns.items():
        gu = name.startswith("g")
        inp = x if gu else a
        ref = fns["gate_up" if gu else "down"](inp, idss[0], w).float(); got = f(inp, idss[0], w).float()
        err = ((ref - got).norm() / ref.norm()).item()
        for i in range(20): f(inp, idss[i], w)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(N_IT): f(inp, idss[i], w)
        g.replay(); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        us = e0.elapsed_time(e1) * 1e3 / N_IT
        byt = uniq * (B_EXP if gu else B_DN)
        print(f"T={T} {name:8s} {us:7.2f} us  {byt/us/1e3:7.0f} GB/s  (unique experts {uniq:.1f}) err {err:.1e}")
