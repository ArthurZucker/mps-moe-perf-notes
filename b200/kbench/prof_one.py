import os, torch
from kernels import get_local_kernel
from pathlib import Path
k = get_local_kernel(Path(os.environ["TRANSFORMERS_MOE_AFFINE_KERNEL"]))
E, H, I, TOPK, G = 512, 2560, 640, 10, 64
wgu = torch.randint(-2**31, 2**31 - 1, (E, 2 * I, H // 8), dtype=torch.int32, device="cuda")
sgu = torch.randn(E, 2 * I, H // G, device="cuda").bfloat16(); bgu = sgu.clone()
x = torch.randn(1, H, device="cuda").bfloat16()
idss = [torch.randperm(E, device="cuda")[:TOPK].int() for _ in range(50)]
fs = {"v1": lambda i: k.gather_qmv4(x, wgu, sgu, bgu, i, TOPK, G), "tc": lambda i: k.gather_qmv4_tc(x, wgu, sgu, bgu, i, TOPK, G)}
for name, f in fs.items():
    for i in idss[:5]: f(i)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for i in idss: f(i)
        torch.cuda.synchronize()
    ev = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    d = sorted(e.device_time for e in ev)
    print(name, "kernels", len(ev), "median us", d[len(d)//2], "min", d[0], set(e.name[:40] for e in ev))
