"""Host-tier handoff in isolation: 48 'layers' in one CUDA graph, each = host_post(n missed routes) -> GPU work of D us
(the resident experts' stand-in) -> host_wait. Random experts from pinned banks (L x EH experts, 2.765 MB each, far
beyond L3) so the CPU streams them from DRAM. Reports us per layer and the CPU-side phase timings."""
import os, sys, time, torch
from kernels import get_local_kernel
from pathlib import Path
k = get_local_kernel(Path(os.environ["TRANSFORMERS_MOE_AFFINE_KERNEL"]))
H, I, G, L, EH, TOPK = 2560, 640, 64, 48, 32, 10
threads = [int(t) for t in os.environ.get("THREADS", "16,32,40").split(",")]
torch.manual_seed(0)
def pinned(*shape, dtype):
    t = torch.empty(*shape, dtype=dtype, pin_memory=True)
    if dtype == torch.int32: t.random_(-2**31, 2**31 - 1)
    else: t.copy_(torch.randn(*shape) * 0.01)
    return t
banks = []
keep = []
for l in range(L):
    gw, gs, gb = pinned(EH, 2 * I, H // 8, dtype=torch.int32), pinned(EH, 2 * I, H // G, dtype=torch.bfloat16), pinned(EH, 2 * I, H // G, dtype=torch.bfloat16)
    dw, ds, db = pinned(EH, H, I // 8, dtype=torch.int32), pinned(EH, H, I // G, dtype=torch.bfloat16), pinned(EH, H, I // G, dtype=torch.bfloat16)
    keep += [gw, gs, gb, dw, ds, db]
    banks += [t.data_ptr() for t in (gw, gs, gb, dw, ds, db)]
print(f"banks {sum(t.numel() * t.element_size() for t in keep) / 1e9:.1f} GB pinned", flush=True)
table = torch.full((L * EH,), -1, dtype=torch.int32, device="cuda")  # nothing resident: every posted route is a miss
state = torch.zeros(3, dtype=torch.long, device="cuda")
hstate = torch.zeros(1, dtype=torch.long, device="cuda")
HINT = int(os.environ.get("HINT", "0")); A = int(os.environ.get("A", "0"))
x = torch.randn(1, H, device="cuda").bfloat16()
node0 = [c for c in range(0, 56)]
V = int(os.environ.get("V", "2"))
start, post, wait, timing, stop = ((k.host_tier_start, k.host_post, k.host_wait, k.host_tier_timing, k.host_tier_stop) if V == 1 else
                                   ((lambda b, h, i, g, c: k.host_tier2_start(b, h, i, g, c, EH)), k.host_post2, k.host_wait2, k.host_tier2_timing, k.host_tier2_stop))
cpu_list = [int(c) for c in os.environ.get("CPUS", ",".join(map(str, node0))).split(",")]
for nt in threads:
    handle, mailbox, result = start(banks, H, I, G, cpu_list[:nt])
    for n in [int(v) for v in os.environ.get("NS", "0,1,2,3,5,10").split(",")]:
        for D in (int(os.environ.get("D", "0")),):
            idss = [torch.randperm(EH, device="cuda")[:max(n, 1)].int() for _ in range(L * 4)]
            ws = torch.rand(max(n, 1), device="cuda")
            out0 = torch.zeros(1, H, device="cuda").bfloat16()
            def step(it):
                o = out0
                for l in range(L):
                    if n and HINT:
                        k.host_hint2(idss[it * L + l], l, EH, table, mailbox, hstate)
                    if A: torch.cuda._sleep(int(A * 1965))  # attention stand-in
                    if n:
                        post(idss[it * L + l], ws, l, *( (EH, table) if V == 1 else (ws * (table[l * EH + idss[it * L + l].long()] >= 0),) ), x, mailbox, state)
                    if D: torch.cuda._sleep(int(D * 1965))  # resident experts stand-in
                    if n:
                        o = wait(o, result, state)
                return o
            torch.cuda.synchronize()
            step(0); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for it in range(4): step(it)
            g.replay(); torch.cuda.synchronize()
            timing(handle)
            best = 1e9
            for _ in range(5):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
                best = min(best, e0.elapsed_time(e1) * 1e3 / (4 * L))
            t = timing(handle)
            c = t[-1]
            cpu = ("cpu " + " ".join(f"{v / c / 1e3:6.1f}" for v in t[:3]) + " us") if c else ""
            print(f"threads {nt:3d} misses {n:2d} hint {HINT} attn {A} res {D}: {best:7.1f} us/layer, extra {best - A - D:6.1f}  {cpu}", flush=True)
    stop(handle)
