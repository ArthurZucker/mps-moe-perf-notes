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

Both on the final Qwen3.8 checkpoint, 16 GiB cache, events off, 32 GB Metal cap.

| transfer | decode | sustained | correctness | verdict |
|---|---:|---:|---|---|
| baseline (no transfer) | 8.56 | 9.10 | 128/128 | reference |
| Inkling GDN/short-conv hub kernels (`use_kernels=True` after load) | 6.94 | 7.90 | 128/128 | **reject: slower** |
| fused Qwen4 router (Metal softmax top-k, one dispatch) | 6.51 | 9.13 | **7/128** | **reject: wrong output** |

The GDN hub kernels are numerically correct but slower on this M3 Ultra; the kernelize path also
needs a post-load workaround because the streamed embed leaves `model.device` on meta. The
fused router kernel is not correct yet (softmax/top-k selection), so it is not pushed as a
perf branch.

## Conclusion

No Inkling kernel transfers positively to Qwen3.8 at 32 GB. The Qwen3.8 gains remain:

- final checkpoint, 0 on-the-fly quantization (published);
- fused sibling linears, 4-bit Metal experts (already in the branch);
- 48 GB for 4-bit (59.59 tok/s) or 2.5-bit Sakura at 30 GB (56-62 tok/s).

The 32 GB 4-bit wall is expert bytes from SSD, not router or GDN dispatch.
