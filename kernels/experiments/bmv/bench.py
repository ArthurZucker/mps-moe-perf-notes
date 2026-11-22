import statistics, torch
lib = torch.mps.compile_shader(open("bmv.metal").read())
CFG = [(32,1,4),(32,2,4),(16,1,4),(16,2,4),(8,1,4),(8,2,4),(32,4,2),(16,4,2),(32,1,8),(16,1,8)]
def t(fn, n):
    for _ in range(3): fn()
    r = []
    for _ in range(7):
        torch.mps.synchronize(); a, e = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
        a.record(); fn(); e.record(); e.synchronize(); r.append(a.elapsed_time(e) / n * 1e3)
    return statistics.median(r)
for N, K in ((1024, 2304), (2304, 3072), (6552, 2304)):
    ws = [torch.randn(N, K, device="mps", dtype=torch.bfloat16) for _ in range(24)]
    x = torch.randn(K, device="mps", dtype=torch.bfloat16); y = torch.empty(N, device="mps", dtype=torch.bfloat16)
    ref = (ws[0].float() @ x.float())
    base = t(lambda: [torch.nn.functional.linear(x[None], w) for w in ws], 24)
    best = None
    for L, R, G in CFG:
        k = getattr(lib, f"bmv_bfloat_{L}_{R}_{G}"); rows_tg = G * (32 // L) * R
        run = lambda: [k(w, x, y, N, K, threads=-(-N // rows_tg) * G * 32, group_size=G * 32) for w in ws]
        k(ws[0], x, y, N, K, threads=-(-N // rows_tg) * G * 32, group_size=G * 32); torch.mps.synchronize()
        err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
        us = t(run, 24)
        best = min(best or (1e9, None), (us, (L, R, G, err)))
    print(f"N={N} K={K}: torch {base:6.1f} us | best bmv {best[0]:6.1f} us {best[1]} ({N * K * 2 / best[0] / 1e3:.0f} GB/s)", flush=True)
