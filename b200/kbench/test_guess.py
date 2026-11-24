import os, ctypes, torch
from kernels import get_local_kernel
from pathlib import Path
k = get_local_kernel(Path(os.environ["TRANSFORMERS_MOE_AFFINE_KERNEL"]))
H, I, E = 2560, 640, 512
banks = [torch.zeros(E, 2 * I, H // 8, dtype=torch.int32, pin_memory=True), torch.zeros(E, 2 * I, H // 64, dtype=torch.bfloat16, pin_memory=True),
         torch.zeros(E, 2 * I, H // 64, dtype=torch.bfloat16, pin_memory=True), torch.zeros(E, H, I // 8, dtype=torch.int32, pin_memory=True),
         torch.zeros(E, H, I // 64, dtype=torch.bfloat16, pin_memory=True), torch.zeros(E, H, I // 64, dtype=torch.bfloat16, pin_memory=True)]
handle, mb, res = k.host_tier2_start([b.data_ptr() for b in banks] * 4, H, I, 64, [5], E)
# Mailbox2 layout: seq 0, hint_seq 8, layer 16, n 20, ntok 24, hint_layer 28, hint_n 32, ids 56, w 312, tok 568, hint_ids 824
L = 4
table = torch.full((L * E,), -1, dtype=torch.int32, device="cuda")
hstate = torch.zeros(1, dtype=torch.long, device="cuda")
scratch = torch.zeros(E, device="cuda"); counter = torch.zeros(1, dtype=torch.int32, device="cuda")
ok = True
for trial in range(20):
    W = torch.randn(E, H, device="cuda").bfloat16(); x = torch.randn(1, H, device="cuda").bfloat16()
    layer = trial % L
    table.fill_(-1); table[layer * E + torch.randperm(E, device="cuda")[:E // 4]] = 0
    kk = 16
    k.host_guess2(x, W, kk, layer, table, mb, hstate, scratch, counter, 64, 8); torch.cuda.synchronize()
    ref_top = (W.float() @ x.float().T).squeeze(1).topk(kk).indices
    ref = [e for e in ref_top.tolist() if table[layer * E + e] < 0]
    n = ctypes.c_int32.from_address(mb + 32).value
    got = [ctypes.c_int32.from_address(mb + 824 + 4 * i).value for i in range(n)]
    hl = ctypes.c_int32.from_address(mb + 28).value
    if set(got) != set(ref) or hl != layer: ok = False; print("MISMATCH", trial, got, ref, hl, layer)
print("guess kernel matches torch topk minus resident:", ok, "| counter reset:", counter.item() == 0)
# timing
x = torch.randn(1, H, device="cuda").bfloat16()
for _ in range(10): k.host_guess2(x, W, 16, 0, table, mb, hstate, scratch, counter, 64, 8)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(100): k.host_guess2(x, W, 16, 0, table, mb, hstate, scratch, counter, 64, 8)
g.replay(); torch.cuda.synchronize()
e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
print(f"host_guess2: {e0.elapsed_time(e1) * 10:.1f} us per call")
k.host_tier2_stop(handle)
