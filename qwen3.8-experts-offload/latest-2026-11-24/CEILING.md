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

## The wall, measured (16 GiB cache, events off, 128/128)

Instrumented `store.stats()` around a 254-token decode (`profile_decode.py`). The three
components sum to the wall exactly, so nothing is hidden:

| component | 16 GiB cold | 16 GiB warm | 40 GiB | 64 GiB |
|---|---:|---:|---:|---:|
| host sync — routing D2H round trip, 48x/token | 39.00 | 32.94 | 34.72 | 32.92 |
| host wait — blocking on expert reads | 42.65 | 8.86 | 20.68 | 5.27 |
| residual — GPU compute + python | 55.58 | 51.80 | 46.50 | 45.02 |
| **wall** | **137.23** | **93.61** | **101.89** | **83.21** |
| tok/s | 7.29 | 10.68 | 9.81 | 12.02 |
| hit rate | 91.2% | 91.8% | 96.9% | 97.0% |

`host sync` is **constant at 32.9-39.0 ms/token from 16 to 64 GiB** — it is a fixed
per-layer latency, not a function of cache size or bytes. `residual` falls 55.58 -> 45.02
as the cache grows, so ~7.6 ms of it is offload-side GPU work (table copies, pass 2); the
remaining ~45 ms is model compute + python.

ms/token. The 16 GiB row sums to 137.23 exactly.

The dominant, *fixable* component is **host sync: one GPU -> host routing round trip per
layer, 48 per token**, worth 35-39 ms/token. It is a fixed per-layer latency, not bytes:
it barely moves when the cache grows (39.00 -> 34.72). Every layer does

    GPU router -> D2H routing copy -> host decides misses -> host reads -> GPU experts

so the host blocks on a pipeline drain 48 times per token, and the expert reads
(42.65 ms) cannot overlap that same layer's GPU work. The 48 serialized round trips are
the wall; the SSD feeds them and is never the limit.

### Per-layer, events off (ms/token, 254 tokens)

The 32.76 ms of sync is spread evenly over all 48 layers, and the 12 `full_attention`
layers (3, 7, 11, ... 47) cost **twice** the 36 `linear_attention` layers:

| layer group | layers | sync ms/token each |
|---|---|---|
| `full_attention` | 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 | 1.05 - 1.08 |
| `linear_attention` | the other 36 | 0.52 - 0.56 |
| all 48 | | sum **32.76** |

Per-layer sum: 32.76 sync + 9.20 read wait = 41.96 ms/token, against the store totals of
32.73 + 9.20. There is no hot layer: the cost is one round trip per layer, 48 times.

Caveat on the read-wait column: these runs had the checkpoint warm in the page cache, so
wait fell from 42.65 (first, cold run) to 9.20 ms/token. Reads matter even less than the
cold number suggested, and the cold number is the honest one for a fresh process.

## Ablations (each outside the ~15% run-to-run noise band)

| ablation | sustained tok/s | change |
|---|---:|---|
| 16 GiB cache, events off (baseline) | 8.2 | — |
| 40 GiB cache, events off (reads 42.65 -> 20.68) | 9.8 (+35%) | reads are real but not dominant |
| 16 GiB cache, **events on** (host work moved off the main thread) | **19.25** | **+135%, 128/128** |

The events-on path (`TRANSFORMERS_EXPERTS_OFFLOAD_MPS_EVENTS=1`) already existed to solve
exactly this: a host thread resolves misses while the device runs pass 1, and the device
waits on an event instead of the host draining the queue. It was previously rejected
because it produced 5/128 — that was the read-ahead eviction bug, fixed today. With
`LOOKAHEAD=0` and the eviction fix it is **128/128 at 19.25 tok/s sustained**, 2.25x the
previous best correct result (8.56) and faster than the old "fastest" 14.41/18.86 that
was broken.

Component split with events on, same harness (`profile_decode.py`, 254 tokens):

| component | events off | events on |
|---|---:|---:|
| main-thread host sync | 39.00 | **0.00** |
| main-thread host wait (reads) | 42.65 | **0.00** |
| GPU compute + python | 55.58 | 55.10 |
| host router thread: spin | — | 33.26 (beside the GPU) |
| host router thread: work | — | 22.21 (beside the GPU) |
| **wall** | **137.23** | **55.10** |
| tok/s | 7.29 | 18.15 |

Events off: the host components are *added* to the GPU time (39.00 + 42.65 + 55.58 =
137.23, exact). Events on: the same 55.47 ms of host work runs *beside* the 55.10 ms of
GPU work, so the wall becomes `max(host, GPU)` instead of `host + GPU`. Host and GPU are
almost perfectly balanced, which is why the gain is 2.5x and why the next lever after
this is reducing both together, not either one alone.

## Not measured: GPU kernel time vs python

`torch.profiler` in torch 2.15 exposes CPU/CUDA/HPU/MTIA/XPU activities but **no MPS
activity**, and `torch.mps.profiler` emits OS signposts that need Instruments/xctrace to
read. So the ~45 ms "GPU + python" residual is reported as one bucket, bounded by the
64 GiB run above, and not split into kernel vs dispatch. That is a tooling gap, not an
unexplored lever: the wall itself (host sync) is already isolated by ablation.


## FINAL LADDER + xctrace verdict (2026-11-25)

Events on, `LOOKAHEAD=0`, `verify_generate.py`, 4g32c, **128/128 on every arm**:

| resident budget | expert cache | sustained tok/s |
|---|---:|---:|
| 32 GB | 16 GiB | 19.58 (no compile) / 18.96 (compile) |
| 48 GB | 32 GiB | 22.47 |
| 64 GB | 48 GiB | 25.11 |
| 96 GB | 80 GiB | 25.31 |

Saturates at 64 GB (all 24,576 expert records resident). Old correct baseline was 8.56 tok/s,
so 32 GB is now **2.29x** and this design's ceiling is ~25 tok/s.

### Instruments 27.2, `Metal System Trace`, exported `mps-hw-intervals`

| window | MPS intervals | GPU busy | GPU occupied | cmd buffers/token |
|---|---:|---:|---:|---:|
| whole run | 1118 | 724 ms | 5% of 15.6 s | - |
| last 1.0 s (decode) | 280 | 151 ms | **15%** | **20** |

586 unnamed command buffers (mean 983 us) + 532 `MPSGraph` (mean 279 us).

**The GPU is ~15% occupied during decode**, and the main thread is ~79% busy with
`sync`/`wait` at 0.00. Three independent GPU-side ablations failed to move the wall
(fused router 18.38, `use_kernels` 17.76, pass-2 zeroed 59.29 vs 56.04 ms control). The wall is
therefore **host-side serialization around ~20 command buffers per token** — not GPU compute,
not SSD, not memory. Tracer perturbs the host, so 15% is an upper bound on GPU occupancy.

### Bottlenecks already excluded by measurement

| hypothesis | how it was killed | result |
|---|---|---|
| SSD bandwidth | cut reads 3.08x (`LOOKAHEAD=0`) | 8.27 -> 8.16, flat; drive does 7.6 GB/s, decode used 1.4 |
| GPU compute | zeroed the whole second expert pass | 59.29 vs 56.04 (no gain; pass 2 is only the ~44 misses) |
| dispatch count | fused router, GDN hub kernels | 18.38 / 17.76 vs 18.96 — slower |
| python/inductor overhead | dropped `torch.compile` entirely | 19.58 vs 18.96 — no loss |
| fullgraph | `fullgraph=True` with setup hoisted + embedding as a custom op | traces, then `kIOGPUCommandBufferCallbackErrorTimeout`: device wait inside one command buffer deadlocks |
| routing copy | timed the copy stage | 0.64 ms/token; `ext.cpu_address` already host-visible |

## What to fix

0. **Ship the events path on.** `TRANSFORMERS_EXPERTS_OFFLOAD_MPS_EVENTS=1` with
   `LOOKAHEAD=0` is 19.25 tok/s at 128/128, 2.25x the events-off path. This is the answer
   to "what is the wall": the per-layer host round trip, removed by overlap.
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


## MAX_AHEAD: the earlier sweep never swept, and the knob saturates at 4

`run_maxahead.sh` exported `TRANSFORMERS_EXPERTS_OFFLOAD_MAX_AHEAD`, but `offload.py` held
`MAX_AHEAD = 16` as a plain class attribute. The 2026-11-25 "48" and "96" runs both ran at 16 and
both ended in `kIOGPUCommandBufferCallbackErrorTimeout`; that timeout was a flaky cold 128-token
verify, not the queue depth.

`benchmark_v2/experts_offload/max_ahead_sweep.py` now sets `MAX_AHEAD` on the live router and
walks values inside one process (one checkpoint load), events on, `LOOKAHEAD=0`, 16 GiB, warm
cache, 16 tokens per arm:

| MAX_AHEAD | ms/token | tok/s |
|---:|---:|---:|
| 1 | 72.25 | 13.84 |
| 2 | 38.09 | 26.25 |
| 4 | 35.94 | 27.83 |
| 8 | 36.03 | 27.75 |
| 12 | 35.65 | 28.05 |
| 16 | 35.75 | 27.97 |
| 24 | 35.86 | 27.89 |
| 32 | 35.67 | 28.04 |

- **Knee at 4.** At 1 the host cannot get the routing before the device needs the table
  (72.25 ms/token, and the hit rate falls to 82% because the host is starved). From 4 to 32
  everything sits inside the ±15% band (35.65-36.03 ms): past the knee the extra depth buys no
  overlap.
- Every arm produced identical tokens; no deadlock up to 32 in this configuration.
- **Cold 128-token verify, MAX_AHEAD=32 with `PIN_AHEAD=1`: 128/128 correct, sustained 17.48**
  against 19.58 for the shipped arm (16, `PIN_AHEAD=0`). The deeper queue does not help, and the
  read-ahead pin costs ~2 tok/s in extra evictions. The default stays 16.
- Warm steady state (99.9% hit) is 35.7 ms/token = **28 tok/s**; the 19.58 ladder number pays the
  cold misses. `ahead_reads` is 0 at that hit rate, so this sweep cannot price read-ahead; it
  only matters in the cold window, where the 32/1 arm above lost.

### Why batching waits (the "two-sided pipeline") cannot remove the handshake

`encodeWaitForEvent` has to run outside an open compute encoder, so every wait costs one encoder
teardown, and the wait must sit between a layer's pass 1 and its pass 2. Layer L+1's pass 1 needs
L's pass-2 result, so K layers cannot share one handshake: `gpu_wait_many(handle, values[], n)`
would batch the *encoding*, not the dependency, and save at most a commit, not the round trip.
The routing copy is 0.63 ms/token and is not the cost.

The per-layer round trip is the cost: ~0.74 ms x 48 = 35.7 ms warm. Removing it needs the experts
resident (48 GiB+ measured 22.47 / 64 GiB 25.11), or a host-side prediction good enough that the
device never needs pass 2 - which is the guess path, not a wait-batching one.


## Instruments, finally: the GPU is 63.5% busy and the host submits every 1.27 ms

Window: events on, LOOKAHEAD=0, MAX_AHEAD=16, 16 GiB, `final-ckpt` 4g32c, 32-token warm-up then 62
measured. Metal System Trace attached to the live decode (12.04 s), table
`metal-application-command-buffer-submissions` exported with `xctrace export`:

| quantity | value |
|---|---:|
| wall | 63.56 ms/token |
| main-thread CPU | 62.79 ms/token (98.8%, `thread_time` verified against `sleep`) |
| command buffers | 9458 / 12.04 s = 785/s = **~50 per token** (~1.04/layer) |
| sum of command-buffer durations | 7.65 s / 12.04 s = **GPU busy 63.5%** |
| command-buffer duration | p50 0.75 ms (p90 1.28, p99 2.98) |
| host gap between submissions | p50 0.94 ms, mean 1.27 |

One command buffer per layer carrying ~0.75 ms of GPU work, submitted every ~1.27 ms: the GPU idles
~0.5 ms per layer waiting for the host. That is the handshake, measured rather than argued. *Earlier
"GPU 15% occupied / 20 buffers per token" was an artefact of a trace that covered only 21.8 s of the
load phase, not decode.*

`route()` is not the cost: copy 0.48 / spin 27.31 / put 0.05 / flush 0.45 ms per token, where the spin
is the MAX_AHEAD backpressure loop, not CPU.

## Four changes, measured, all neutral or worse
| change | result |
|---|---|
| `NO_COMPILE=1` | CPU 58.2 vs 62.8, wall 64.1 vs 63.6 -> neutral |
| route() without `torch.cat` | 63.56 vs 63.62 -> neutral |
| skip `lengths.tolist()` in the 1-sequence rotary split | 63.56 vs 63.62 -> neutral (that call is 0.4 ms/token) |
| `GUESS=1` (decoupled from `lookahead`) | 70.22 vs 63.56 -> 10% worse |

## The guess pass was never running
`guess()` is gated on `next_layers()`, i.e. on `lookahead`, which is 0 for the shipped path. Every arm
therefore predicted nothing (`guess_calls 3072, no next layer 3065`). Decoupled with `ahead_layer()` +
`GUESS=1`: 2952 comparable layers, **2.4% exact**, ~69% per expert, 4525 ahead reads (+53%), 10% slower.
Predicting layer L+1 from layer L's *input* is the wrong predictor: the correlation is per expert and
the wrong experts are the reads you pay for. The right predictor, if any, is temporal (layer L's
experts at token t-1), which needs no extra reads because those slots are already resident.


## CORRECTION: the MAX_AHEAD knee and the "28 tok/s warm steady state" were a 16-token artefact

The sweep above measured 16 tokens per arm straight after a 32-token warm-up, where the 16 GiB cache is at
**99.9% hit**. That is not a steady state: over 256 tokens on the same prompt the hit rate settles at **92%**
and the same arms land 40% lower.

MAX_AHEAD sweep, 256 tokens per arm, one prompt, events on, LOOKAHEAD=0, single process:

| MAX_AHEAD | 2 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ms/token | 53.94 | 53.76 | 54.97 | 54.95 | 54.00 | 56.14 | 53.60 |
| tok/s | 18.54 | 18.60 | 18.19 | 18.20 | 18.52 | 17.81 | 18.66 |
| hit | 92.4% | 92.5% | 92.4% | 92.2% | 92.6% | 91.8% | 92.6% |

- **Flat from 2 to 32** (18.2-18.7 tok/s, inside noise). The earlier "knee at 4" was an artefact of the short
  warm window; only `MAX_AHEAD=1` is genuinely bad (it starves the host).
- The earlier "warm steady state 28 tok/s / 35.7 ms per token" is **withdrawn**. 32/48/64/96 GB ladder numbers and
  the 17.62 sustained in the 128-token verify are the ones to quote.
- Arms are not bit-identical to each other (`tokens DIFFERENT`), so cross-arm token comparison is not an exactness
  test here; the eager-vs-compiled gate in `verify_generate.py` is (128/128 on the pushed tree).
