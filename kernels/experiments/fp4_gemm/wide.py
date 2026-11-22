import sys, statistics, torch
sys.path.insert(0, "."); import fp4_gemm as G
H, I, E, R = 2304, 1152, 128, 8
torch.manual_seed(0)
for fmt in (0, 1):
    if fmt == 0:
        w = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="mps"); s = (torch.rand(E, 2 * I, H // 16, device="mps") * 0.4 + 0.05).to(torch.float8_e4m3fn).view(torch.uint8); f = torch.rand(E, 2, device="mps")
    else:
        w = torch.randint(0, 256, (E, 2 * I, H), dtype=torch.uint8, device="mps"); w = torch.where((w & 0x7F) == 0x7F, w - 1, w); s = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, device="mps"); f = torch.ones(1, device="mps")
    for T in (512, 2048):
        x = torch.randn(T, H, device="mps", dtype=torch.bfloat16)
        _, ids = torch.topk(torch.rand(T, E, device="mps"), R, dim=-1)
        bm = 32
        perm, te, tr, re_, mt = G._tiles(ids, E, bm)
        lib = G._lib(); a1 = torch.empty(T * R, I, dtype=x.dtype, device="mps"); a2 = torch.empty_like(a1)
        k1 = getattr(lib, f"gemm_gate_up_{fmt}_bfloat_{bm}"); k2 = getattr(lib, f"gemm_gate_up_wide_{fmt}_bfloat_{bm}")
        r1 = lambda: k1(w, s, f, x, perm, te, tr, re_, a1, I, H, R, 4.0, 25.0, threads=[mt * 128, I // 32], group_size=[128, 1])
        r2 = lambda: k2(w, s, f, x, perm, te, tr, re_, a2, I, H, R, 4.0, 25.0, threads=[mt * 256, I // 64], group_size=[256, 1])
        r1(); r2(); torch.mps.synchronize()
        err = ((a1.float() - a2.float()).abs().max() / a1.float().abs().max()).item()
        def t(fn):
            for _ in range(2): fn()
            res = []
            for _ in range(7):
                torch.mps.synchronize(); a, e = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
                a.record(); fn(); e.record(); e.synchronize(); res.append(a.elapsed_time(e))
            return statistics.median(res)
        fl = 2 * T * R * 2 * I * H
        t1, t2 = t(r1), t(r2)
        print(f"fmt {fmt} T={T}: gate_up {t1:6.2f} ms ({fl / t1 / 1e9:4.1f} TF) -> wide {t2:6.2f} ms ({fl / t2 / 1e9:4.1f} TF), diff {err:.1e}", flush=True)
