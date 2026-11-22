import sys, statistics, torch
sys.path.insert(0, "."); import prefill_attention as PA
torch.manual_seed(0)
H, HKV, D, R = 24, 8, 128, 1024
def reference(q, k, v, rs, proj, dist, sinks, scaling, window):
    rel = (rs @ proj).transpose(1, 2)  # [1, H, T, R]
    idx = dist.clamp(0, R - 1).expand(*rel.shape[:2], -1, -1)
    bias = rel.gather(-1, idx).masked_fill((dist < 0) | (dist >= R), 0.0)
    valid = (dist >= 0) & ((dist < window) if window else True)
    kk = k.repeat_interleave(H // HKV, 1); vv = v.repeat_interleave(H // HKV, 1)
    w = (q @ kk.transpose(2, 3)) * scaling + bias.to(q.dtype)
    w = w.masked_fill(~valid, torch.finfo(w.dtype).min)
    s = sinks.transpose(1, 2).unsqueeze(-1).to(w.dtype)
    p = torch.softmax(torch.cat([w, s], -1).float(), -1)[..., :-1].to(q.dtype)
    return (p @ vv).transpose(1, 2)
def t(fn):
    fn(); torch.mps.synchronize(); r = []
    for _ in range(5):
        a, e = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
        torch.mps.synchronize(); a.record(); fn(); e.record(); e.synchronize(); r.append(a.elapsed_time(e))
    return statistics.median(r)
for T in (512, 777, 2048):
    L = -(-T // 32) * 32
    q = torch.randn(1, H, T, D, device="mps", dtype=torch.bfloat16)
    k = torch.randn(1, HKV, L, D, device="mps", dtype=torch.bfloat16); v = torch.randn(1, HKV, L, D, device="mps", dtype=torch.bfloat16)
    rs = torch.randn(1, T, H, 16, device="mps") * 0.5; proj = torch.randn(16, R, device="mps") * 0.3
    sinks = torch.randn(1, T, H, device="mps")
    dist = torch.arange(T, device="mps")[:, None] - torch.arange(L, device="mps")[None, :]
    for window in (1024, None):
        args = (q, k, v, rs, proj, dist, sinks, D ** -0.5, window)
        a = reference(*args).float(); b = PA.prefill_attention(*args).float()
        err = ((a - b).abs().max() / a.abs().max()).item()
        print(f"T={T:5d} window={window}: rel err {err:.1e} | eager {t(lambda: reference(*args)):7.2f} ms | kernel {t(lambda: PA.prefill_attention(*args)):6.2f} ms", flush=True)
