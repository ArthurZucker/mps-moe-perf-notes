import statistics, sys, torch
base = open("base.metal").read()
H = 2304
def bench(tgs):
    src = base.replace("constant constexpr uint MG_TGS = 16;", f"constant constexpr uint MG_TGS = {tgs};")
    lib = torch.mps.compile_shader(src)
    k = getattr(lib, "message_gate_bfloat_0_2304")
    split = tgs * 32 // 64
    flags = torch.zeros(tgs, dtype=torch.int32, device="mps"); ex = torch.zeros(64 * split + tgs, device="mps")
    torch.manual_seed(0)
    sets = [(torch.randn(H, device="mps"), torch.randn(H, device="mps"), torch.randn(64, 2 * H, device="mps", dtype=torch.bfloat16) * 0.02,
             torch.randn(H, 64, device="mps") * 0.05, torch.randn(H, 64, device="mps") * 0.05) for _ in range(24)]
    outs = [torch.empty(H, device="mps") for _ in sets]
    run = lambda: [k(h, m, d, f, w, d, o, flags, ex, 1e-6, 0.0, threads=tgs * 1024, group_size=1024) for (h, m, d, f, w), o in zip(sets, outs)]
    run(); torch.mps.synchronize(); ref = [o.clone() for o in outs]
    r = []
    for _ in range(7):
        torch.mps.synchronize(); a, e = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
        a.record(); run(); e.record(); e.synchronize(); r.append(a.elapsed_time(e) / 24 * 1e3)
    return statistics.median(r), ref
t16, r16 = bench(16); t24, r24 = bench(24)
err = max(((a - b).abs().max() / a.abs().max()).item() for a, b in zip(r16, r24))
print(f"MG_TGS=16: {t16:.1f} us | MG_TGS=24: {t24:.1f} us | rel diff {err:.1e}")
