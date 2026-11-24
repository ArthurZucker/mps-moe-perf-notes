# What transfers from Inkling-2-Mini to Qwen3.8-Flash-Next

## Inkling final checkpoint (published)

`ArthurZ/Inkling-2-Mini-metal-final` (commit `65ab95b`): dense linears packed offline,
native NVFP4/MXFP8 experts unchanged.

- load: **0** on-the-fly quantization calls
- driver after `empty_cache`: **17.8 GiB**; footprint **18 GB**; peak **18 GB**
  (parent + runtime plan: 24.9 GiB / 18 GB / **37 GB**)
- `hc_bench`, `base-4bit`, 6x128, M3 Ultra: **71.39 tok/s** (71.5 / 71.4 / 71.35 /
  71.42 / 71.36 / 71.38), `looping_output: false`
- parent + runtime plan: 79.14 tok/s

The 37 GB peak was the runtime `MetalConfig(4,64)` plan packing 201 bf16 linears at load.
The final checkpoint moves that work offline. Qwen3.8 already has the same treatment in
`ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit-final`.

## Transfer list

| Inkling lever | Qwen3.8 status | Action |
|---|---|---|
| prequantized dense final checkpoint | **done** | 725 linears packed, 0 calls |
| `fuse_sibling_linears` | **done** | 192 groups fused |
| `qmv4` / `qmv4_multi` decode matvec | **done** | stream-experts MetalLinear |
| fused routed-expert decode (`qmv4`/`experts_down_sum`) | **done** | `MetalExperts._fused_decode` |
| fused router top-k (`router_topk`, one dispatch) | **not in Qwen3.8** | candidate: Qwen4 router uses `torch.topk`; porting the fused sigmoid/top-k/route kernel is generic |
| GDN / short-conv Metal kernels (`linear_attention.py`) | **not enabled by default** | candidate: Qwen3.8 GDN uses the `causal_conv1d` fallback unless `use_kernels=True`; the Inkling `use_kernels=True` test produced garbage, so this needs an exactness gate per kernel |
| `down_route` fixed-order expert sum | Qwen3.8 uses the fused decode sum | already covered |
| `fp4_gate_up_act` / `fp4_gemm` (NVFP4/MXFP8) | not applicable | Qwen3.8 experts are affine 4-bit |
| `inkling_attention` / `prefill_attention` | not applicable | Qwen3.8 uses QSA, not relative-bias + sinks |
| `last_loop` KV | not applicable | Qwen3.8 has no looped layer stack |

## Highest-value transfer

1. **Fused Qwen4 router**: one `router_topk` dispatch instead of matmul + softmax +
   topk + gather/normalize. This is the same shape class as Inkling's router, low risk,
   and does not touch the expert I/O bottleneck.
2. **GDN/conv Metal kernels**: test each kernel against torch for Qwen3.8 before enabling.
   `use_kernels=True` alone is not safe: it gave 38.1 tok/s and garbage on Inkling's
   public branch.
3. **Expert I/O remains the Qwen3.8 wall**: at 32 GB the 4-bit expert bank holds 9,122 of
   24,576 records; 30.88 tok/s (JITEngine). No Inkling kernel changes that. 48 GB gives
   59.59 tok/s; 2.5-bit Sakura gives 56-62 at 30 GB.

## Not transferable

The Inkling 79 tok/s result is mostly a **capacity** result: 18 GB fits. Qwen3.8's 65 GB
expert bank does not. The same final-checkpoint and fusion work is already applied; the
remaining gap is bytes per token from SSD, not dispatch count.

## Measured transfer attempts on Qwen3.8

All on the final Qwen3.8 checkpoint, 16 GiB expert cache, events off, 32 GB Metal cap,
128/128 greedy tokens matching eager in every arm below.

The earlier revision of this table rejected both kernels, and the router was reported at
7/128. That verdict does not survive: the read-ahead path in `offload.py` could evict a
resident expert of the layer it was prefetching for while the GPU was still in that
layer's first pass, so a slot could be read as the wrong expert. With read-ahead disabled
(`TRANSFORMERS_EXPERTS_OFFLOAD_PIN_AHEAD` unset, now the default) every arm is 128/128.

| transfer | decode | sustained | correctness | verdict |
|---|---:|---:|---|---|
| router off, kernels off | 6.54 / 6.73 | 8.27 / 7.20 | 128/128 | reference |
| fused Qwen4 router | 6.78 / 6.98 | 8.31 / 8.24 | 128/128 | **exact, neutral perf** |
| fused router + GDN hub kernels | 6.77 / 6.71 | 8.97 / 7.24 | 128/128 | **exact, perf inconclusive** |

The two numbers per cell are separate runs. Run-to-run spread on this machine is ~15%
(load average ~2, 35 days uptime), and the original 8.56/9.10 baseline no longer
reproduces on its own script: re-running `run_verify16_events_off.sh` unchanged gives
6.73/7.20. So no arm here shows a delta outside noise, and the earlier "slower" and
"wrong" verdicts were both artefacts: one of the eviction bug, one of thermal drift.

What is solid:

- The fused router is numerically exact. Its previous 7/128 was the eviction bug, not
  the softmax/top-k math. Logits are stored as bf16-rounded doubles and the full softmax
  denominator is used, because Qwen4 sets `norm_topk_prob=None` (no renormalisation).
- The GDN/short-conv hub kernels are exact for Qwen3.8 and load without the meta-device
  failure when enabled after load (`model.set_use_kernels(True)`); the `use_kernels=True`
  constructor path still dies on `Unsupported device type 'meta'`.
- Neither lever moves the 32 GB number. The router is one dispatch per layer against a
  ~37 ms/token host sync and multi-GB/token expert reads; the GDN kernels are a few
  small ops per layer. At 16 GiB cache the wall is expert bytes from SSD.

## Conclusion

No Inkling kernel transfers a measurable win to Qwen3.8 at 32 GB. What did transfer is a
correctness fix: the read-ahead eviction bug in the Qwen3.8 offload path, which corrupted
expert slots and made the two candidate kernels look wrong or slow.

The Qwen3.8 gains remain:

- final checkpoint, 0 on-the-fly quantization (published);
- fused sibling linears, 4-bit Metal experts (already in the branch);
- fused Qwen4 router and GDN hub kernels, both exact and both pushed on
  `qwen3.8-metal-perf` (perf-neutral, kept because they are correct and reduce dispatch);
- 48 GB for 4-bit (59.59 tok/s) or 2.5-bit Sakura at 30 GB (56-62 tok/s).

The 32 GB 4-bit wall is expert bytes from SSD, not router or GDN dispatch.
