import os, torch
from kernels import get_local_kernel
from pathlib import Path
k = get_local_kernel(Path(os.environ["TRANSFORMERS_MOE_AFFINE_KERNEL"]))
E, H = 512, 2560
banks = [torch.zeros(E, 1280, 320, dtype=torch.int32, pin_memory=True), torch.zeros(E, 1280, 40, dtype=torch.bfloat16, pin_memory=True),
         torch.zeros(E, 1280, 40, dtype=torch.bfloat16, pin_memory=True), torch.zeros(E, 2560, 80, dtype=torch.int32, pin_memory=True),
         torch.zeros(E, 2560, 10, dtype=torch.bfloat16, pin_memory=True), torch.zeros(E, 2560, 10, dtype=torch.bfloat16, pin_memory=True)]
handle, mb, res = k.host_tier2_start([b.data_ptr() for b in banks], H, 640, 64, [5], E)
table = torch.full((E,), -1, dtype=torch.int32, device="cuda"); hstate = torch.zeros(1, dtype=torch.long, device="cuda")
scratch = torch.zeros(E, device="cuda"); counter = torch.zeros(1, dtype=torch.int32, device="cuda")
W = torch.randn(E, H, device="cuda").bfloat16(); x = torch.randn(1, H, device="cuda").bfloat16()
for name, mo, blocks in (("logits only, 64 blocks", -1, 64), ("full, 64 blocks", 4, 64), ("logits only, 128 blocks", -1, 128), ("full, 128", 4, 128)):
    f = lambda: k.host_guess2(x, W, 16, 0, table, mb, hstate, scratch, counter, mo, blocks)
    for _ in range(5): f()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(50): f()
    g.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
    print(f"{name}: {e0.elapsed_time(e1) * 20:.1f} us")
k.host_tier2_stop(handle)
