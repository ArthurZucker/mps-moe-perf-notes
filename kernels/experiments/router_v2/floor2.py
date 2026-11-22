import time, statistics, torch
lib = torch.mps.compile_shader("""
#include <metal_stdlib>
using namespace metal;
kernel void plus1(device float* x, uint i [[thread_position_in_grid]]) { x[i] += 1.0f; }
kernel void fenced(device float* x, device atomic_uint* c, uint tid [[thread_index_in_threadgroup]], uint n [[threadgroups_per_grid]]) {
    x[tid] += 1.0f;
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) { atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
                    atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);
                    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); }
}
""")
x = torch.zeros(4096, device="mps"); c = torch.zeros(1, dtype=torch.int32, device="mps")
def timed(fn, n=48):
    for _ in range(5): fn()
    ts = []
    for _ in range(9):
        torch.mps.synchronize(); t = time.perf_counter(); fn(); torch.mps.synchronize(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) / n * 1e6
print(f"x.add_(1)                  {timed(lambda: [x.add_(1) for _ in range(48)]):6.1f} us")
print(f"compile_shader plus1       {timed(lambda: [lib.plus1(x, threads=4096, group_size=256) for _ in range(48)]):6.1f} us")
print(f"compile_shader fenced 16tg {timed(lambda: [lib.fenced(x, c, threads=16*256, group_size=256) for _ in range(48)]):6.1f} us")
print(f"plus1 + 3 empty allocs     {timed(lambda: [(torch.empty(1,128,device='mps'), torch.empty(1,8,device='mps'), torch.empty(1,8,dtype=torch.int64,device='mps'), lib.plus1(x, threads=4096, group_size=256)) for _ in range(48)]):6.1f} us")
