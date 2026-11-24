import torch, sys
from pathlib import Path
from kernels import get_local_kernel
k = get_local_kernel(Path(sys.argv[1]))
torch.manual_seed(0)
def gt(f, it=50):
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10): f()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    g.replay(); torch.cuda.synchronize(); s.record()
    for _ in range(it): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it / 10 * 1e3
# distinct weights per call so we measure HBM, not L2: rotate over enough copies to exceed L2 (126 MB)
for (N, K) in [(10240, 2560), (6144, 2560), (2560, 6144), (12288, 2560), (320, 10240), (10240, 320), (640, 2560), (2560, 640), (4, 10240), (48, 2560), (248320, 2560)]:
    copies = max(1, int(400e6 // (N * K * 2)))
    Ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
    for M in (1, 4):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x.float(), Ws[0].float())
        got = k.gemv_bf16(x, Ws[0]).float()
        err = ((got - ref).norm() / ref.norm()).item()
        it = iter(range(10**9))
        def kf(): return k.gemv_bf16(x, Ws[next(it) % copies])
        it2 = iter(range(10**9))
        def tf(): return torch.nn.functional.linear(x, Ws[next(it2) % copies])
        # graphs capture fixed pointers: build a graph that cycles copies explicitly
        def cyc(fn):
            def f():
                for W in Ws[:min(copies, 10)]: fn(W)
            return f
        n = min(copies, 10)
        us_k = gt(cyc(lambda W: k.gemv_bf16(x, W))) / n
        us_t = gt(cyc(lambda W: torch.nn.functional.linear(x, W))) / n
        by = N * K * 2
        print(f"N={N:6d} K={K:5d} M={M}  err {err:.1e}  kernel {us_k:7.1f} us {by/us_k/1e3:5.0f} GB/s | cuBLAS {us_t:7.1f} us {by/us_t/1e3:5.0f} GB/s")
