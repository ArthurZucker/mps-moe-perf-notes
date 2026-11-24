"""Per-layer extra cost with the host tier off the critical path: the post / poll / result read run on a side stream
concurrently with the resident experts (main stream joins before the add). L=48 layers, attention A us, resident D us."""
import os, torch
from kernels import get_local_kernel
from pathlib import Path
k = get_local_kernel(Path(os.environ["TRANSFORMERS_MOE_AFFINE_KERNEL"]))
H, I, G, L, EH = 2560, 640, 64, 48, 32
A, D = int(os.environ.get("A", "100")), int(os.environ.get("D", "25"))
def pinned(*shape, dtype):
    t = torch.empty(*shape, dtype=dtype, pin_memory=True)
    if dtype == torch.int32: t.random_(-2**31, 2**31 - 1)
    else: t.copy_(torch.randn(*shape) * 0.01)
    return t
keep, banks = [], []
for l in range(L):
    b = (pinned(EH, 2 * I, H // 8, dtype=torch.int32), pinned(EH, 2 * I, H // G, dtype=torch.bfloat16), pinned(EH, 2 * I, H // G, dtype=torch.bfloat16),
         pinned(EH, H, I // 8, dtype=torch.int32), pinned(EH, H, I // G, dtype=torch.bfloat16), pinned(EH, H, I // G, dtype=torch.bfloat16))
    keep += b; banks += [t.data_ptr() for t in b]
handle, mb, res = k.host_tier2_start(banks, H, I, G, list(range(1, 56)), EH)
table = torch.full((L * EH,), -1, dtype=torch.int32, device="cuda")
state = torch.zeros(2, dtype=torch.long, device="cuda"); hstate = torch.zeros(1, dtype=torch.long, device="cuda")
x = torch.randn(1, H, device="cuda").bfloat16(); zero = torch.zeros(1, H, device="cuda").bfloat16()
side = torch.cuda.Stream()
for hint in (0, 1):
    for n in (0, 1, 2, 3, 5, 10):
        idss = [torch.randperm(EH, device="cuda")[:max(n, 1)].int() for _ in range(L * 2)]
        w = torch.rand(max(n, 1), device="cuda")
        def step(it):
            main = torch.cuda.current_stream()
            o = zero
            for l in range(L):
                ids = idss[(it * L + l) % len(idss)]
                if n and hint:
                    ev0 = torch.cuda.Event(); ev0.record(main)
                    with torch.cuda.stream(side):
                        side.wait_event(ev0); k.host_hint2(ids, l, EH, table, mb, hstate)
                torch.cuda._sleep(A * 1965)  # attention + router
                if n:
                    ev_r = torch.cuda.Event(); ev_r.record(main)
                    with torch.cuda.stream(side):
                        side.wait_event(ev_r)
                        k.host_post2(ids, w, l, w * (table[l * EH + ids.long()] >= 0), x, mb, state)
                        hres = k.host_wait2(zero, res, state)
                        ev_d = torch.cuda.Event(); ev_d.record(side)
                torch.cuda._sleep(D * 1965)  # resident experts
                if n:
                    main.wait_event(ev_d)
                    o = o + hres
            return o
        step(0); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for it in range(2): step(it)
        g.replay(); torch.cuda.synchronize()
        k.host_tier2_timing(handle)
        best = 1e9
        for _ in range(5):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            best = min(best, e0.elapsed_time(e1) * 1e3 / (2 * L))
        t = k.host_tier2_timing(handle)
        cpu = ("cpu " + " ".join(f"{v / t[-1] / 1e3:5.1f}" for v in t[:3])) if t[-1] else ""
        print(f"side-stream hint {hint} misses {n:2d}: {best:6.1f} us/layer, extra {best - A - D:5.1f}  {cpu}", flush=True)
k.host_tier2_stop(handle)
