import sys, time, torch
sys.path.insert(0, "/Users/arthurzucker/Work/hub-kernels/metal-moe-decode/build/torch-metal")
import router as R
lib = torch.mps.compile_shader("""
#include <metal_stdlib>
using namespace metal;
kernel void plus1(device float* x, uint i [[thread_position_in_grid]]) { x[i] += 1.0f; }""")
xx = torch.zeros(4096, device="mps")
w = torch.randn(8, 64, device="mps"); b = torch.zeros(8, device="mps"); x = torch.randn(1, 64, device="mps", dtype=torch.bfloat16)
for name, fn in [("router E8 H64", lambda: R._router_mps(x, w, b, 1, 1.0)), ("plus1", lambda: lib.plus1(xx, threads=4096, group_size=256))]:
    for _ in range(50): fn()
    torch.mps.synchronize()
    s, e = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
    t = time.perf_counter(); s.record()
    for _ in range(480): fn()
    e.record(); cpu = (time.perf_counter() - t) / 480 * 1e6; e.synchronize()
    wall = (time.perf_counter() - t) / 480 * 1e6
    print(f"{name:14s} cpu enqueue {cpu:5.1f} us | wall {wall:5.1f} us | gpu events {s.elapsed_time(e)/480*1e3:5.1f} us", flush=True)
