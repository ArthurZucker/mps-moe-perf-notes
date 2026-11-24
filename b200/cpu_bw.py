"""CPU DRAM read bandwidth with many threads (numpy-free: torch int32 bitwise ops over a 4-bit expert-shaped bank),
plus a 4-bit-expert-matvec proxy: how long do 70 x 2.76 MB experts take to read and multiply on the host?"""
import torch, time, os
for threads in (28, 56, 112):
    torch.set_num_threads(threads)
    x = torch.randint(0, 2**31 - 1, (1 << 28,), dtype=torch.int32)  # 1 GiB
    y = torch.empty_like(x)
    torch.bitwise_and(x, 0x0F0F0F0F, out=y)
    t = time.perf_counter()
    for _ in range(5): torch.bitwise_and(x, 0x0F0F0F0F, out=y)
    dt = (time.perf_counter() - t) / 5
    print(f"{threads:3d} threads: read+write {2 * x.numel() * 4 / dt / 1e9:6.1f} GB/s", flush=True)
# matvec proxy: dequant-free bf16 GEMV over the same bytes as 70 experts' 4-bit banks (2.76 MB each -> 193 MB bf16-equivalent bytes read)
torch.set_num_threads(112)
W = torch.randn(70 * 1280, 1080, dtype=torch.bfloat16)  # ~193 MB
v = torch.randn(1080, dtype=torch.bfloat16)
torch.mv(W, v); t = time.perf_counter()
for _ in range(5): torch.mv(W, v)
dt = (time.perf_counter() - t) / 5
print(f"bf16 GEMV over 193 MB (70 experts' worth of bytes): {dt * 1e3:.2f} ms -> {W.numel() * 2 / dt / 1e9:.1f} GB/s")
