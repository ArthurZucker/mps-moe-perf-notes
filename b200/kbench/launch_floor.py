import torch
x = torch.zeros(1, device="cuda"); y = torch.zeros(1 << 20, device="cuda")
def run(f, n=500):
    for _ in range(3): f()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): f()
    g.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / n
print("tiny add_ (1 elem)   us/kernel", round(run(lambda: x.add_(1)), 2))
print("add_ 1M elems (4MB)  us/kernel", round(run(lambda: y.add_(1)), 2))
