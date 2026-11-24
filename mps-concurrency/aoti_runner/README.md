# AOTI command-block runner: one concurrent encoder, explicit barriers

This is the AOTI counterpart of `multistream.py`: instead of multiple queues, it hooks
the AOTI `aoti_torch_mps_run_command_block` path and encodes every Metal kernel into one
`MTLDispatchTypeConcurrent` compute encoder. The runner tracks the `MTLBuffer` set by each
kernel (`arg 0` output, the rest inputs) and inserts
`memoryBarrierWithResources` only for RAW/WAR/WAW edges.

No mega-kernel. The generated AOTI kernels are unchanged.

## Files

- `concurrent_runner.mm`: overrides the `aoti_torch_mps_*` shim symbols linked into the
  AOTI wrapper. Exposes `mps_concurrent_begin/finish/set_conservative`.
- `patch_aoti_pkg.py`: extracts a `.pt2`, rewrites the wrapper to begin/finish around the
  Metal command blocks, compiles the wrapper with the runner, repackages.
- `toy_aoti.py`, `toy_dep.py`: small AOTI models (pure Metal, and mixed Metal + cumsum).
- `compare_toy.py`, `compare_dep.py`: serial `.pt2` vs patched `.pt2`, exactness + timing.

## Results (M3 Ultra, torch 2.15)

Pure Metal graph: two independent chains, 1M elements, 8 kernels fused to 2 Metal kernels:

| build | ms | exact |
|---|---:|---|
| serial AOTI | 0.317 | yes |
| concurrent runner | **0.286** | yes |

Mixed graph: `sin -> cumsum -> cos` per chain, 65,536 elements:

| build | ms | exact vs eager |
|---|---:|---|
| serial AOTI | 0.323 | no (maxdiff 7.6e-6, cumsum math) |
| concurrent runner | 1.777 | **yes** |

The mixed case is correct but slower: `aoti_torch_mps_cumsum` runs on PyTorch's stream,
so the runner flushes/waits at every mixed-op boundary. The concurrent encoder only pays
off on a run of Metal kernels with no PyTorch-stream ops in between.

## How it works

- `aoti_torch_mps_start_encoding` sets the `MetalKernelFunction`'s encoder to our
  concurrent encoder.
- `aoti_torch_mps_set_arg_tensor` records the `MTLBuffer` (`idx == 0` write, rest read).
- `aoti_torch_mps_dispatch_*` inserts `memoryBarrierWithResources` if a read was written
  after the last barrier, or a write was read/written after it.
- `run_command_block` executes the generated lambda; `finish` ends/commits/waits.

## Limitations

- Assumes the first tensor arg is the output (true for the generated wrappers tested).
- Defines an empty `__binary_constants_bin_start/end`; a model with a constants blob must
  link the original blob instead.
- Mixed PyTorch-stream ops need a flush; otherwise they race the separate queue.
- This is a concurrent **compute encoder**, not `MTLIndirectCommandBuffer` yet. The
  command-block hook is the same; ICB needs an indirect-command encoder for the set_args
  and dispatch calls.

## Run

```bash
# compile a toy AOTI package
python toy_aoti.py
# patch it (builds concurrent_runner.mm and repackages)
python patch_aoti_pkg.py toy.pt2 toy_concurrent.pt2
# compare
python compare_toy.py
```

## Follow-up: replacing the PyTorch-stream cumsum with a Metal kernel

`aoti_torch_mps_cumsum` was the mixed op that forced a flush. The runner now defines it
in Metal (three-stage scan for n > 1024, sequential for n <= 1024) and dispatches it into
the same concurrent encoder, so no PyTorch stream sync.

| graph | build | ms | maxdiff vs eager |
|---|---|---:|---:|
| 2 independent Metal kernels | serial | 0.317 | 0 |
| | concurrent | 0.286 | 0 |
| 8 independent Metal kernels (65K each) | serial | 0.294 | 0 |
| | concurrent | 0.340 | 0 |
| sin+Metal cumsum+cos (65K) | serial AOTI | 0.381 | 7.6e-6 |
| | concurrent Metal cumsum | 0.546 | 4.6e-5 / 1.5e-4 |

The custom cumsum removes the PyTorch block, but it is **slower** than PyTorch's own
cumsum at 65K and not bit-identical. Replacing a PyTorch op only helps if the custom Metal
kernel itself is faster; otherwise the runner overhead + scan cost wins.

## 8 independent kernels

The concurrent encoder did not beat the serial AOTI graph for eight 65K-element kernels:
the kernels are large enough to keep the GPU busy, while the runner pays command-buffer
and submission overhead. The concurrency win is for many small kernels or graphs with
real independent branches; 1170-kernel Inkling decode has a critical path of 915, so the
single-stream ceiling from concurrency alone is still ~3-5%.

## What to replace, in order

1. Ops that force a stream flush **and** are cheap to implement as a Metal kernel
   (pointwise/norm/indexing), not scan/linalg.
2. Ops on the critical path of the Inkling AOTI wrapper, not every PyTorch op.
3. Only keep a replacement if it beats the PyTorch op in-context; cumsum did not.
