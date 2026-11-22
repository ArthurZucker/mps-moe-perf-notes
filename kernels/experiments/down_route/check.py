import sys, torch
sys.path.insert(0, "/Users/arthurzucker/Work/hub-kernels/metal-moe-decode/build/torch-metal")
import fp4_experts_opt as K
H, I, E, R = 2304, 1152, 128, 8
for fmt in (0, 1):
    if fmt == 0:
        w = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device="mps"); s = (torch.rand(E, H, I // 16, device="mps") * 0.4 + 0.05).to(torch.float8_e4m3fn); f = torch.rand(E, device="mps")
    else:
        w = torch.randint(0, 256, (E, H, I), dtype=torch.uint8, device="mps"); w = torch.where((w & 0x7F) == 0x7F, w - 1, w).view(torch.float8_e4m3fn); s = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, device="mps"); f = torch.ones(1, device="mps")
    for T in (1, 3):
        wts, ids = torch.topk(torch.rand(T, E, device="mps"), R, dim=-1)
        wts = wts.to(torch.bfloat16)
        act = torch.randn(T * R, I, device="mps", dtype=torch.bfloat16)
        a = K.down_sum(act, w, s, f, ids, wts).float()
        b = K.down_route(act, w, s, f, ids, wts).view(T, R, H).sum(1)
        print(fmt, T, "rel err", ((a - b).abs().max() / a.abs().max()).item())
