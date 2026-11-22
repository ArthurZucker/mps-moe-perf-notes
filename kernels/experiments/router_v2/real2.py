import sys, statistics, torch
sys.path.insert(0, "."); import router as new
torch.manual_seed(0)
banks = [(torch.randn(128, 2304, device="mps") * 0.02, torch.randn(128, device="mps") * 0.01) for _ in range(48)]
x = torch.randn(1, 2304, device="mps", dtype=torch.bfloat16)
cnt = torch.zeros(1, dtype=torch.int32, device="mps")
lo, ow, oi = torch.empty(1, 128, device="mps"), torch.empty(1, 8, device="mps"), torch.empty(1, 8, dtype=torch.int64, device="mps")
src = new._SOURCE
libs = {"full": torch.mps.compile_shader(src),
        "no selection": torch.mps.compile_shader(src.replace("    if (!is_last || sg != 0) return;", "    return;")),
        "no last-tg sync": torch.mps.compile_shader(src.replace("""        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        is_last = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed) == n_tg - 1;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);""", "        is_last = tg == 0;"))}
for name, lib in libs.items():
    k = getattr(lib, "router_topk_bfloat_8")
    fn = lambda: [k(x, w, b, lo, ow, oi, cnt, 128, 2304, 8, 2.5, threads=128 * 256, group_size=256) for w, b in banks]
    for _ in range(3): fn()
    res = []
    for _ in range(7):
        torch.mps.synchronize(); s, e = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize(); res.append(s.elapsed_time(e) / 48 * 1e3)
    print(f"{name:16s} {statistics.median(res):5.1f} us/call", flush=True)
