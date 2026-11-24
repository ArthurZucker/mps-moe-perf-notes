# Actual 32 GB ceiling: overlap, hit rate, miss bandwidth

The earlier claim that 32 GB + 50 tok/s is impossible was wrong. The measured 32 GB
step time decomposes into compute plus serialized I/O; with proper overlap the ceiling is
above 50.

## Inputs

- 48 MoE layers, top-10 routing: **480 expert routes/token**.
- One 4-bit expert record (gate_up + down + bf16 scales/biases): **2.7648 MB**.
- All-resident compute/host floor: 64 GiB JITEngine = 69.80 tok/s = **14.33 ms/token**.
- Measured 32 GiB JITEngine: 30.88 tok/s = **32.4 ms/token**.
- Difference 32.4 - 14.3 = **18.1 ms/token** of I/O.
- If that I/O is 48 misses/token (90% hit), 48 x 2.7648 = 132.7 MB, so the miss path
  is running at ~**7.3 GB/s**. That matches the internal SSD.

So the 32 GB run is approximately **compute + I/O serialized**, not overlapped:
`14.3 + 18.1 = 32.4 ms`. The overlap opportunity is 1.8x.

## Ceiling

`step = max(compute, misses x 2.7648 MB / SSD)`, 480 routes/token.

| hit rate | misses/token | MB/token | at 5 GB/s | at 7.3 GB/s | at 10 GB/s |
|---:|---:|---:|---:|---:|---:|
| 80% | 96 | 265.4 | 18.8 | 27.5 | 37.7 |
| 85% | 72 | 199.1 | 25.1 | 36.7 | 50.2 |
| **90%** | **48** | **132.7** | 37.7 | **55.0** | 69.8 |
| 95% | 24 | 66.4 | 69.8 | 69.8 | 69.8 |

(tok/s, compute floor 14.33 ms; I/O below the floor is hidden.)

**50 tok/s at 32 GB requires ~90% hit at 7.3 GB/s, or ~85% hit at 10 GB/s, with full
overlap.** The current 32 GB run already looks like ~90% hit but serializes the 18.1 ms
of I/O behind the 14.3 ms of compute. Fixing overlap is the lever, not more memory.

At 80% hit, 32 GB tops out near 27-38 tok/s even with perfect overlap and a 7-10 GB/s
SSD. At 95% hit, the 14.3 ms compute floor dominates and 32 GB can reach the same
~70 tok/s as the 64 GB run.

## What to fix

1. Overlap the miss read with the next layer's compute, not the current layer's wait.
   The JITEngine notes already describe the right shape: GPU pass 1 on resident experts,
   host reads misses, next-layer router runs on the provisional output.
2. Do not let read-ahead evict a slot the running layer may read. See
   `patches/0001-offload-pin-ahead-and-lookahead.diff`.
3. Treat a non-miss whose slot changed owner as a miss in `_resolve` (epoch/ownership
   check), or pin the in-flight layer until pass 2 completes.
4. A super-fast miss is worth ~1.5-2x at 32 GB; a larger cache buys the rest.
