import os, torch
from kernels import get_local_kernel
from pathlib import Path
k = get_local_kernel(Path(os.environ["TRANSFORMERS_MOE_AFFINE_KERNEL"]))
H, I, G, L, EH, TOPK = 2560, 640, 64, 4, 32, 10
def pinned(*shape, dtype):
    t = torch.empty(*shape, dtype=dtype, pin_memory=True)
    if dtype == torch.int32: t.random_(-2**31, 2**31 - 1)
    else: t.copy_(torch.randn(*shape) * 0.01)
    return t
keep, banks = [], []
for l in range(L):
    b = (pinned(EH, 2 * I, H // 8, dtype=torch.int32), pinned(EH, 2 * I, H // G, dtype=torch.bfloat16), pinned(EH, 2 * I, H // G, dtype=torch.bfloat16),
         pinned(EH, H, I // 8, dtype=torch.int32), pinned(EH, H, I // G, dtype=torch.bfloat16), pinned(EH, H, I // G, dtype=torch.bfloat16))
    keep += b; banks += [t.data_ptr() for t in b]
handle, mb, res = k.host_tier2_start(banks, H, I, G, list(range(1, 56)), EH)
table = torch.full((L * EH,), -1, dtype=torch.int32, device="cuda")
state = torch.zeros(3, dtype=torch.long, device="cuda"); hstate = torch.zeros(1, dtype=torch.long, device="cuda")
x = torch.randn(1, H, device="cuda").bfloat16(); out = torch.zeros(1, H, device="cuda").bfloat16()
n = int(os.environ.get("N", "1"))
idss = [torch.randperm(EH, device="cuda")[:n].int() for _ in range(64)]
w = torch.rand(n, device="cuda")
def step():
    o = out
    for i in range(32):
        l = i % L
        if os.environ.get("HINT") == "1": k.host_hint2(idss[i], l, EH, table, mb, hstate)
        torch.cuda._sleep(100 * 1965)
        k.host_post2(idss[i], w, l, w * (table[l * EH + idss[i].long()] >= 0), x, mb, state)
        torch.cuda._sleep(25 * 1965)
        o = k.host_wait2(o, res, state)
    return o
step(); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): step()
g.replay(); torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
    g.replay(); torch.cuda.synchronize()
import collections
d = collections.defaultdict(list)
for e in p.events():
    if e.device_type == torch.autograd.DeviceType.CUDA: d[e.name[:40]].append(e.device_time)
for kname, v in d.items():
    v.sort(); print(f"{kname:42s} n={len(v):3d} median {v[len(v)//2]:7.2f} us  min {v[0]:7.2f}")
# gaps: total graph time vs sum of kernels
e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
print(f"graph {e0.elapsed_time(e1) * 1e3 / 32:.1f} us/layer; kernel sum/layer {sum(sum(v) for v in d.values()) / 32:.1f}")
t = k.host_tier2_timing(handle); print("cpu us", [round(v / t[-1] / 1e3, 1) for v in t[:3]])
k.host_tier2_stop(handle)
