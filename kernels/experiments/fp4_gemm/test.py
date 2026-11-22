"""experts_prefill (grouped GEMM) vs fp4_experts_opt.experts_decode (reference matvec kernels): correctness + time."""
import sys, time, statistics, torch
sys.path.insert(0, "/Users/arthurzucker/Work/hub-kernels/metal-moe-decode/build/torch-metal")
import fp4_experts_opt as ref
import fp4_gemm as G
H, I, E, R = 2304, 1152, 128, 8
torch.manual_seed(0)
def bank(fmt, rows, cols, gu):
    if fmt == "nvfp4":
        w = torch.randint(0, 256, (E, rows, cols // 2), dtype=torch.uint8, device="mps")
        s = (torch.rand(E, rows, cols // 16, device="mps") * 0.4 + 0.05).to(torch.float8_e4m3fn)
        f = (torch.rand(E, 2 if gu else 1, device="mps") * 0.1 + 0.02).squeeze(-1)
    else:
        w = torch.randint(0, 256, (E, rows, cols), dtype=torch.uint8, device="mps")
        w = torch.where((w & 0x7F) == 0x7F, w - 1, w).view(torch.float8_e4m3fn)
        s = torch.randint(118, 124, (E, rows, cols // 32), dtype=torch.uint8, device="mps")
        f = torch.ones(E, device="mps")
    return w, s, f
def timed(fn):
    fn(); torch.mps.synchronize(); ts = []
    for _ in range(5):
        t = time.perf_counter(); fn(); torch.mps.synchronize(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3
for fmt in ("nvfp4", "mxfp8"):
    gu, dn = bank(fmt, 2 * I, H, True), bank(fmt, H, I, False)
    for T in (512, 2048, 4096):
        x = torch.randn(T, H, device="mps", dtype=torch.bfloat16) * 0.5
        logits = torch.randn(T, E, device="mps")
        wts, ids = torch.topk(torch.sigmoid(logits), R, dim=-1)
        args = (x, *gu, *dn, ids, wts, 4.0, 25.0)
        a = ref.experts_decode(*args).float(); b = G.experts_prefill(*args).float()
        err = ((a - b).abs().max() / a.abs().max()).item()
        tr, tg = timed(lambda: ref.experts_decode(*args)), timed(lambda: G.experts_prefill(*args))
        flops = 2 * T * R * (2 * I * H + H * I)
        print(f"{fmt} T={T:5d}: rel err {err:.2e} | matvec {tr:7.2f} ms | gemm {tg:7.2f} ms ({flops / tg / 1e9:5.2f} TFLOPS) | x{tr / tg:5.1f}", flush=True)
