# Multi-stream results on torch 2.15 (M3 Ultra)

`multistream.py` uses real `torch.mps.Stream` queues and explicit events. In torch 2.15
the Python `Event.record/wait` only allows the default stream, but the C++ calls
(`torch._C._mps_recordEvent/_mps_waitForEvent`) already use the current stream; the
harness bypasses the stale Python guard. Outputs are verified bit-exact against serial.

## Results

Two independent dependent chains, 500 steps, M3 Ultra, torch 2.15:

| size | serial 2 branches | 2 streams | 4 streams | 8 streams |
|---:|---:|---:|---:|---:|
| 4,096 | 8.88 ms | 9.24 ms (1.05x) | 19.01 ms | 34.07 ms |
| 65,536 | 9.94 ms | 9.83 ms (0.98x) | 18.31 ms | 37.27 ms |
| 1,048,576 | 20.98 ms | 21.74 ms (0.96x) | 41.83 ms | 84.06 ms |

All outputs exact. Throughput is flat as branches are added: the extra streams do not
overlap on this stack. The 100-step run showed the same: 2 streams 0.91x on 1 MiB.

## Interpretation

`torch.mps.Stream` gives separate queues and events, but eager kernels do not overlap
across them on torch 2.15. The measured concurrency win in `ISSUE_DRAFT.md` is from a
**concurrent compute encoder inside one command buffer**, not from multiple queues.

PyTorch's `MPSStream::commandEncoder()` still creates a serial encoder. The one-line
concurrent-encoder PR (pytorch/pytorch#189785) was closed as a draft because Metal does
not maintain ordering between concurrent dispatches without explicit resource barriers;
the author moved to multi-stream support instead. pytorch/pytorch#205181/#205743 add
stream/event APIs, but they do not make this eager workload overlap.

There is no public indirect-command-buffer API in `MPSStream`; a whole-model command
buffer with indirect dispatch would need a PyTorch MPS backend change, not a model patch.

## Consequence

To use concurrent dispatch for Inkling without mega-kernels, the MPS backend has to
expose either:

1. a concurrent compute encoder plus per-buffer dependency tracking / explicit
   `memoryBarrierWithResources`, or
2. an indirect command buffer / AOTI runner that accepts a prebuilt command buffer.

The model's decode graph is mostly a dependent chain (1170 kernels, critical path 915 in
the notes), so even a perfect concurrent encoder can only recover ~3-5% for single-stream
decode. The larger wins are batched requests, prefill branches, and independent expert
groups.
