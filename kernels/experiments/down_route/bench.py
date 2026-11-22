import sys, statistics, torch
sys.path.insert(0, "/Users/arthurzucker/Work/hub-kernels/metal-moe-decode/build/torch-metal")
import fp4_experts_opt as ref
lib = torch.mps.compile_shader(open("down_route.metal").read())
H, I, E, R = 2304, 1152, 128, 8
torch.manual_seed(0)
def bank(fmt):
    if fmt == 0:
        return (torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device="mps"), (torch.rand(E, H, I // 16, device="mps") * 0.4 + 0.05).to(torch.float8_e4m3fn).view(torch.uint8), torch.rand(E, device="mps") * 0.1 + 0.02)
    w = torch.randint(0, 256, (E, H, I), dtype=torch.uint8, device="mps"); w = torch.where((w & 0x7F) == 0x7F, w - 1, w)
    return (w, torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, device="mps"), torch.ones(E, device="mps"))
CFG = [(8,1,16,4),(8,2,16,4),(8,4,16,4),(16,1,16,4),(16,2,16,4),(32,1,16,4),(4,1,16,4),(8,2,16,8),(8,1,32,4),(4,2,32,4),(8,2,8,4)]
def t(fn, n):
    for _ in range(3): fn()
    res = []
    for _ in range(9):
        torch.mps.synchronize(); a, e = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
        a.record(); fn(); e.record(); e.synchronize(); res.append(a.elapsed_time(e) / n * 1e3)
    return statistics.median(res)
for fmt in (0, 1):
    w, s, f = bank(fmt)
    calls = []
    for _ in range(16):  # distinct routing per call: DRAM reads
        wts, ids = torch.topk(torch.rand(1, E, device="mps"), R, dim=-1)
        act = torch.randn(R, I, device="mps", dtype=torch.bfloat16)
        calls.append((ids.int().contiguous(), wts.float().contiguous(), act))
    wref = (w if fmt == 0 else w.view(torch.float8_e4m3fn), s if fmt == 0 else s, f)
    ids, wts, act = calls[0]
    y_ref = ref.down_sum(act, wref[0], wref[1].view(torch.float8_e4m3fn) if fmt == 0 else wref[1], f, ids, wts).float()
    base = t(lambda: [ref.down_sum(a, wref[0], wref[1].view(torch.float8_e4m3fn) if fmt == 0 else wref[1], f, i, ww) for i, ww, a in calls], 16)
    print(f"fmt {fmt}: down_sum {base:5.1f} us", flush=True)
    for (lpr, rpl, vpl, sg) in CFG:
        k = getattr(lib, f"down_route_{fmt}_{lpr}_{rpl}_{vpl}_{sg}")
        rows_tg = (32 // lpr) * rpl * sg
        part = torch.empty(R, H, device="mps")
        def run(i, ww, a):
            k(w, s, f, a, i, ww, part, H, I, 0, threads=R * (H // rows_tg) * sg * 32, group_size=sg * 32)
            return part.sum(0)
        y = run(ids, wts, act)
        err = ((y - y_ref).abs().max() / y_ref.abs().max()).item()
        us = t(lambda: [run(i, ww, a) for i, ww, a in calls], 16)
        print(f"   lpr={lpr:2d} rpl={rpl} vpl={vpl:2d} sg={sg}: {us:5.1f} us (incl. sum) rel err {err:.1e}", flush=True)
