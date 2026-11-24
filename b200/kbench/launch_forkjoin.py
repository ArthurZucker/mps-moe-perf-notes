"""cudaGraphLaunch CPU cost: a 48-layer graph of ~35 small kernels per layer, plain vs with a per-layer side-stream fork/join
(the host tier's structure: record on main, side waits, 2 kernels on side, record, main waits) and a second fork."""
import time, torch
x = torch.zeros(1024, device="cuda")
side, bg = torch.cuda.Stream(), torch.cuda.Stream()
def build(mode):
    def step():
        main = torch.cuda.current_stream()
        for l in range(48):
            for _ in range(33): x.add_(1)
            if mode >= 1:
                e0 = torch.cuda.Event(); e0.record(main)
                with torch.cuda.stream(side):
                    side.wait_event(e0); x.mul_(1); x.mul_(1); e1 = torch.cuda.Event(); e1.record(side)
            if mode >= 2:
                with torch.cuda.stream(bg):
                    bg.wait_event(e0); x.sub_(0); x.sub_(0)
                    e2 = torch.cuda.Event(); e2.record(bg)
            for _ in range(2): x.add_(1)
            if mode >= 1: main.wait_event(e1)
        if mode >= 2: main.wait_event(e2)
    step(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): step()
    return g
for mode, name in ((0, "plain"), (1, "+ side fork/join per layer"), (2, "+ side + bg fork per layer")):
    g = build(mode); g.replay(); torch.cuda.synchronize()
    ls, ws = [], []
    for _ in range(20):
        torch.cuda.synchronize(); t0 = time.perf_counter(); g.replay(); t1 = time.perf_counter(); torch.cuda.synchronize(); t2 = time.perf_counter()
        ls.append(t1 - t0); ws.append(t2 - t0)
    ls.sort(); ws.sort()
    print(f"{name:30s}: launch (CPU) {ls[10] * 1e3:6.2f} ms, launch+run {ws[10] * 1e3:6.2f} ms")
