# Upstream patch needed: MPSStream concurrent encoder + explicit barriers

Status: design, not built. This is the exact backend change the multi-stream result points to.
The installed torch 2.15 and nightly 2.16 both serialize eager kernels across streams; the
microbench shows a concurrent encoder can overlap independent groups, but only with barriers.

## File: `aten/src/ATen/mps/MPSStream.h`

Add:

```cpp
  // A concurrent encoder (MTLDispatchTypeConcurrent). The caller owns ordering:
  // insert barrier() / barrierWithResources() between dependent dispatches.
  MTLComputeCommandEncoder_t commandEncoderConcurrent();
  void barrier(MTLBarrierScope scope = MTLBarrierScopeBuffers);
  void barrierWithResources(NSArray<id<MTLResource>>* resources);
```

Private:

```cpp
  bool _enableConcurrentEncoder = false;
```

## File: `aten/src/ATen/mps/MPSStream.mm`

```cpp
MTLComputeCommandEncoder_t MPSStream::commandEncoderConcurrent() {
  if (!_commandEncoder) {
    _commandEncoder = [[commandBuffer()
        computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] retain];
  }
  return _commandEncoder;
}

void MPSStream::barrier(MTLBarrierScope scope) {
  dispatch_sync(_serialQueue, ^() {
    [commandEncoderConcurrent() memoryBarrierWithScope:scope];
  });
}

void MPSStream::barrierWithResources(NSArray<id<MTLResource>>* resources) {
  dispatch_sync(_serialQueue, ^() {
    [commandEncoderConcurrent() memoryBarrierWithResources:resources
                                                    count:resources.count];
  });
}
```

`commandEncoder()` stays serial by default. A separate opt-in entry point is required
because every existing caller assumes implicit ordering.

## Required codegen change

`torch._inductor.codegen.mps` / the Metal wrapper must emit:

- one `commandEncoderConcurrent()`;
- a `barrierWithResources` for every read-after-write / write-after-read /
  write-after-write edge between kernels in the same encoder;
- `barrier()` at true graph boundaries.

The dependency graph is already available in Inductor's scheduler. The notes
`graph_width.py` parses the generated wrapper and builds the same read/write DAG
(1170 kernels, critical path 915 for Inkling decode). Without this, enabling the
concurrent encoder is unsafe: Metal does not maintain ordering.

## Expected result

From `ISSUE_DRAFT.md`: independent tiny kernels 1.56-3.44x; independent matvecs
1.33-1.66x; dependent chains no gain. Inkling decode is a mostly dependent chain
(critical path 915/1170), so the single-stream ceiling from concurrency is only ~3-5%.
The wins are batched requests, prefill branches, and independent expert groups.

## Existing upstream state

- `pytorch/pytorch#189785` (one-line concurrent encoder): closed as a draft; unsafe without
  resource ordering.
- `pytorch/pytorch#205181`, `#205743`: MPS stream/event APIs; these do not expose a concurrent
  encoder.
- `pytorch/pytorch#206006`: host-signalled events / host pointers; separate from encoder
  concurrency.
- No public indirect-command-buffer API in `MPSStream`.

## Verification plan after a source build

1. `bench.py` independent groups: serial vs concurrent + barriers, 2/3/8 groups.
2. `multistream.py` exact-output check against serial.
3. `graph_width.py` critical path; expected decode gain <=5% for single stream.
4. Inkling `hc_bench` A/B: control / concurrent encoder / control, 6 x 128 tokens,
   idle GPU, exact greedy text.
