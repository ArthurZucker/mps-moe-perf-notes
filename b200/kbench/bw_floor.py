"""What does reading R random experts' gate_up codes cost with no compute? index_select copy (read + write) per call,
back-to-back in a CUDA graph, vs one big contiguous copy."""
import torch
E, H, I = 512, 2560, 640
w = torch.randint(-2**31, 2**31 - 1, (E, 2 * I, H // 8), dtype=torch.int32, device="cuda")
for R in (10, 20, 30, 40, 80, 160):
    N_IT = 200
    idss = [torch.randperm(E, device="cuda")[:R] for _ in range(N_IT)]
    buf = torch.empty(R, 2 * I, H // 8, dtype=torch.int32, device="cuda")
    for i in range(5): torch.index_select(w, 0, idss[i], out=buf)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(N_IT): torch.index_select(w, 0, idss[i], out=buf)
    g.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
    us = e0.elapsed_time(e1) * 1e3 / N_IT
    byt = buf.numel() * 4
    print(f"R={R:4d} read {byt/1e6:6.1f} MB  {us:7.2f} us  read {byt/us/1e3:6.0f} GB/s (read+write {2*byt/us/1e3:6.0f})")
