# [MPS] Concurrent compute encoder with graph-derived barriers (and fewer encoder breaks around MPSGraph ops)

### 🚀 The feature, motivation and pitch

`MPSStream` always encodes kernels into a serial compute encoder (`aten/src/ATen/mps/MPSStream.mm`,
`[commandBuffer() computeCommandEncoder]`, i.e. `MTLDispatchTypeSerial`). With a serial encoder Metal makes every
dispatch wait for the previous one, even when the two share no buffers. Metal can do better:
`computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent` plus explicit `memoryBarrierWithScope:` /
`memoryBarrierWithResources:` only where data actually flows (Metal 4's `MTL4ComputeCommandEncoder` is concurrent by
default with explicit barriers). Inductor already knows the dependencies of the kernels it emits, so it could pass them down
as barriers. Separately, every MPSGraph op (`mm`, `addmm`, ...) ends kernel coalescing and opens a new encoder.

Measured on an M3 Max (30-core GPU), macOS 27, torch 2.15. Microbenchmark: ~250 lines of Objective-C++ against torch's
`MTLDevice` (own command buffers, GPU timestamps); source in this folder (`mps_concurrency.mm`, `bench.py`).

**Independent kernels** (groups of G dispatches with no dependency, 1200 dispatches per command buffer):

| kernel | G | serial encoder | concurrent encoder + barrier between groups | speedup |
|---|---:|---:|---:|---:|
| tiny (4096 threads) | 2 | 1.15 us/dispatch | 0.73 | x1.56 |
| tiny | 3 | 1.15 | 0.61 | x1.87 |
| tiny | 8 | 1.15 | 0.33 | x3.44 |
| matvec 1024 x 2304 bf16 | 3 | 7.16 | 5.36 | x1.33 |
| matvec 1024 x 2304 bf16 | 8 | 7.37 | 4.44 | x1.66 |

**Dependent chains** gain nothing (serial 1.39 vs concurrent + barrier 1.49 us/dispatch, tiny; 6.29 vs 6.52 matvec):
this request is about not serializing independent work, not about removing true dependencies.

**Encoder breaks:** a new encoder per dispatch costs +1.5 us per matvec (6.29 -> 7.77 us). On an exact (bf16) MoE
decode graph we see 209-353 MPSGraph `mm` ops per token, each one an encoder break.
[E14 result here: the same decode with every batch-1 linear as a custom kernel, i.e. no MPSGraph breaks.]

**Where it matters (and where not):** we measured the dependency DAG of a real compiled decode graph (Inkling-2-Mini, 48 layer
evaluations per token, parsed from inductor's wrapper, read-after-write only): 1170 kernels, critical path 915, so a concurrent
encoder would recover only ~3-5% for single-stream decode of this model. The gains are for wide graphs: batched / multi-request
decode, MoE layers dispatching many independent experts, prefill with parallel branches, and graphs mixing MPSGraph ops with
custom kernels. `graph_width.py` computes the width of any inductor wrapper.

**Proposal**
1. `MPSStream` option (or default) to open concurrent compute encoders; `MPSStream` / inductor's Metal codegen emit
   `memoryBarrierWithResources:` from the buffers a kernel reads that an earlier kernel in the same encoder wrote
   (inductor has this in its scheduler; eager could use per-buffer last-writer tracking).
2. Keep custom-shader / inductor kernels in the same encoder across MPSGraph ops where possible, or batch MPSGraph ops,
   to cut encoder churn.

### Alternatives

Fusing independent kernels by hand (e.g. one dispatch doing three projections with different inputs) works per model but
doesn't scale. Grid-synchronized "megakernels" hit co-residency limits (deadlock once the grid exceeds the resident
threadgroups). A software dependent launch (the next kernel loads its weights, then spins on a device flag) only helps the
tiniest kernels (x1.63 at 74 KB weights) and loses beyond (x0.40-0.79): spinning threadgroups starve the producer.

### Additional context

Related: #206006 (host-signalled events: the host <-> GPU side of avoiding serialization in the same `MPSStream`).
Full notes and data: https://github.com/ArthurZucker/mps-moe-perf-notes

cc @kulinseth @malfet @DenisVieriu97 @jhavukainen @aditvenk @Isalia20


## Follow-up (2026-11-24): multi-stream does not overlap eager kernels on torch 2.15

`multistream.py` forks real `torch.mps.Stream` queues with explicit events (C++ event ops;
the Python guard only allows the default stream). Two dependent chains at 4K/64K/1M:
2 streams = 0.96-1.05x, 4 streams = ~0.5x, 8 streams = ~0.25x, outputs exact. So separate
queues do not overlap; the concurrent-encoder path above is still the lever, and it needs
the per-resource barrier design (or indirect command buffers) that pytorch#189785 lacked.


Nightly 2.16.0.dev20261123 with the new `Event.record(stream)` / `wait(stream)` API:
2 streams 1.00x at 4K and 0.96x at 1M, 4 streams 0.50x, 8 streams 0.25x, exact. The
event API is not the blocker; separate streams still serialize eager kernels.
