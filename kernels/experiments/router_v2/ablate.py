import sys, re, time, torch
sys.path.insert(0, "/Users/arthurzucker/Work/hub-kernels/metal-moe-decode/build/torch-metal")
import router as R
base = R._SOURCE
variants = {
  "full": base,
  "no selection": base.replace("    if (!is_last || sg != 0) return;", "    return;"),
  "no atomic/fence": base.replace("""        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        is_last = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed) == n_tg - 1;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);""", "        is_last = true;"),
  "no mem_device barrier": base.replace("    threadgroup_barrier(mem_flags::mem_device);\n    threadgroup bool", "    threadgroup bool").replace(
        "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);", "threadgroup_barrier(mem_flags::mem_threadgroup);"),
}
w = torch.randn(8, 64, device="mps"); b = torch.zeros(8, device="mps"); x = torch.randn(1, 64, device="mps", dtype=torch.bfloat16)
cnt = torch.zeros(1, dtype=torch.int32, device="mps")
lo, ow, oi = torch.empty(1, 8, device="mps"), torch.empty(1, 8, device="mps"), torch.empty(1, 8, dtype=torch.int64, device="mps")
for name, src in variants.items():
    assert name == "full" or src != base, name
    k = getattr(torch.mps.compile_shader(src), "router_topk_bfloat")
    fn = lambda: k(x, w, b, lo, ow, oi, cnt, 8, 64, 1, 1.0, threads=256, group_size=256)
    for _ in range(50): fn()
    torch.mps.synchronize(); s, e = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
    s.record()
    for _ in range(480): fn()
    e.record(); e.synchronize()
    print(f"{name:22s} gpu {s.elapsed_time(e)/480*1e3:5.1f} us", flush=True)
