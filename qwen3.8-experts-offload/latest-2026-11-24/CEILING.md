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

## Measured: I/O is not the ceiling (2026-11-24, 16 GiB cache, events off)

The read-ahead path was reading 2.96 GB/token when the all-miss cost is 1.33 GB/token:
speculative reads for the next layer were being thrown away when the guess was wrong.
Setting `TRANSFORMERS_EXPERTS_OFFLOAD_LOOKAHEAD=0` cuts disk reads **3.08x**:

| arm | sustained | decode mean | disk read | bytes/token | read rate |
|---|---:|---:|---:|---:|---:|
| lookahead 1 (default) | 8.27 | 6.73 | 187.7 GiB | 2.96 GB | 1.43 GB/s |
| lookahead 0 | 8.16 | 5.76 | 60.9 GiB | 1.01 GB | 0.49 GB/s |

**Throughput did not move.** Removing two thirds of the reads left sustained decode flat
and the mean slightly worse. So the 32 GB run is not I/O-bound, and the internal SSD is
not the ceiling: `dd` on the checkpoint reads at **7.6 GB/s**, and decode never exceeds
1.4 GB/s of it. The stall model in the first half of this file over-weights I/O.

What the wall actually is: **host + GPU work per token.** At 16 GiB the measured non-stall
cost is ~24.8 ms/token against ~8.5 ms of I/O stall, and the 48 GiB run's non-stall cost
is ~16.8 ms/token. The 8 ms/token difference between 32 and 48 GiB is host cache
bookkeeping, not SSD reads. The router and GDN kernels are dispatch-count reductions, and
they were perf-neutral, so dispatch count is not the wall either.

The unmeasured part is where the ~24.8 ms of non-stall time goes: GPU kernels, host sync,
and the offload table bookkeeping. That needs a host-side profile before more kernel work.

## What to fix

1. Overlap the miss read with the next layer's compute, not the current layer's wait.
   The JITEngine notes already describe the right shape: GPU pass 1 on resident experts,
   host reads misses, next-layer router runs on the provisional output.
2. Do not let read-ahead evict a slot the running layer may read. See
   `patches/0001-offload-pin-ahead-and-lookahead.diff`.
3. Treat a non-miss whose slot changed owner as a miss in `_resolve` (epoch/ownership
   check), or pin the in-flight layer until pass 2 completes.
4. A super-fast miss is worth ~1.5-2x at 32 GB; a larger cache buys the rest.


## Measured correction (32 GB /stats)

The measured 32 GB run is **96.8% hit**, not 80-90%:

- 37.2 MB/token miss bytes, 8.5 ms/token I/O stall
- 24.8 ms/token non-stall (host + GPU)
- 33.3 ms/token wall = 30 tok/s
- perfect overlap of the I/O alone: `1000/24.8 = 40.3 tok/s`

So the simple overlap model gives ~40 tok/s, not 55. The 48 GiB run reaches 59.6 tok/s
because its non-stall cost is ~16.8 ms/token. The remaining 8 ms/token at 32 GB is host
cache bookkeeping / MTP verify work, not SSD reads. To reach 50 at 32 GB, that host work
must be reduced or overlapped; more cache is the easier answer (48 GiB).
