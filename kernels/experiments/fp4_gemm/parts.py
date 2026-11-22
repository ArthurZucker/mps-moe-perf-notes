import sys, time, statistics, torch
import fp4_gemm as G
H, I, E, R = 2304, 1152, 128, 8
torch.manual_seed(0)
gu = (torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="mps"), (torch.rand(E, 2 * I, H // 16, device="mps") * 0.4 + 0.05).to(torch.float8_e4m3fn), torch.rand(E, 2, device="mps"))
dn = (torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device="mps"), (torch.rand(E, H, I // 16, device="mps") * 0.4 + 0.05).to(torch.float8_e4m3fn), torch.rand(E, device="mps"))
def timed(fn):
    fn(); torch.mps.synchronize(); ts = []
    for _ in range(7):
        t = time.perf_counter(); fn(); torch.mps.synchronize(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3
for T in (16, 512):
    x = torch.randn(T, H, device="mps", dtype=torch.bfloat16)
    wts, ids = torch.topk(torch.rand(T, E, device="mps"), R, dim=-1)
    perm, te, tr, re_, mt = G._tiles(ids, E)
    lib = G._lib()
    act = torch.empty(T * R, I, dtype=x.dtype, device="mps"); y = torch.empty(T * R, H, device="mps")
    gate = lambda: getattr(lib, "gemm_gate_up_0_bfloat")(gu[0], gu[1].view(torch.uint8), gu[2], x, perm, te, tr, re_, act, I, H, R, 4.0, 25.0, threads=[mt * 128, I // 32], group_size=[128, 1])
    down = lambda: getattr(lib, "gemm_down_0_bfloat")(dn[0], dn[1].view(torch.uint8), dn[2], act, perm, wts.view(torch.uint8), te, tr, re_, y, H, I, 0, threads=[mt * 128, H // 32], group_size=[128, 1])
    def unperm():
        out = torch.empty_like(y); out[perm] = y; return out.view(T, R, H).sum(1).to(x.dtype)
    print(f"T={T}: tiles {timed(lambda: G._tiles(ids, E)):.2f} ms | gate_up {timed(gate):.2f} | down {timed(down):.2f} | unpermute {timed(unperm):.2f}", flush=True)
