import torch, sys
from pathlib import Path
from kernels import get_local_kernel
k = get_local_kernel(Path("/home/arthur/mps-moe-perf-notes/b200/kernel-src/moe-affine"))
E = 2764800  # bytes per expert (gate_up + down + scales/biases)
NE = 64
host = torch.empty(NE * E, dtype=torch.uint8).pin_memory(); host.fill_(1)
dev = torch.empty(NE * E, dtype=torch.uint8, device="cuda")
def t_graph(f, it=30):
    f(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    g.replay(); torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(it): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1e3
for n in (1, 2, 4, 10):
    us = t_graph(lambda: dev[: n * E].copy_(host[: n * E], non_blocking=True))
    print(f"DMA memcpy {n:2d} experts: {us:7.1f} us  {n*E/us/1e3:5.1f} GB/s")
# SM copy: install kernel (one bank = the whole expert as a flat [NE, E] uint8 bank)
pool = dev.view(NE, E)
for n in (1, 2, 4, 10):
    ids = torch.arange(n, device="cuda", dtype=torch.int32); slots = ids.clone(); miss = torch.ones_like(ids)
    for blocks in (16, 64, 148, 296):
        us = t_graph(lambda: k.offload_install(ids, slots, miss, [host.data_ptr()], [pool], blocks))
        print(f"SM install {n:2d} experts, {blocks:3d} blocks/expert: {us:7.1f} us  {n*E/us/1e3:5.1f} GB/s")
