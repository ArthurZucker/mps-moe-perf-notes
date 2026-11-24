---
license: apache-2.0
tags:
- kernels
- cuda
- moe
---

# moe-affine

CUDA kernels for small-batch (decode, 1-4 tokens) mixture-of-experts layers whose experts are stored in the
**affine 4-bit `MetalExperts` / MLX layout**: codes LSB-first in `uint32` words, per-group (32/64/128) bf16 scales and
biases, `w = scale * code + bias`. Used by `transformers` (`integrations/cuda_affine_moe.py`) to run e.g.
`ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit` on CUDA without dequantizing the experts.

## Ops

| op | what |
|---|---|
| `qmv4_tc6(x, pool, ids, top_k, x_div, group_size, host=None, slots=None, miss=None, weights=None)` | tensor-core expert matvec (group size 64; gate/up K=2560 and down K=640 with <= 10 routes). Per block: one cp.async batch stages a 16-row tile of every routed expert + scales + activations, codes are dequantized as bf16 `128 + c` with one `lop3` per pair, `mma.m16n8k16`. With `host/slots/miss` it reads missed experts from pinned host memory (UVA) and installs them into the device pool. |
| `gather_qmv4`, `down_route4` | scalar (FMA) expert matvecs, any group size: per-route gate/up outputs, and down fused with the routing-weighted sum |
| `gemv_bf16(x, W, bias)` | bf16 GEMV for 1-4 (or 8) rows, short outputs |
| `router_topk(x, W, k, norm)` | router logits + softmax + top-k (+ renorm) in one launch, up to 4 tokens |
| `offload_plan`, `qmv4_offload`, `qmv4_offload_v2`, `offload_install`, `offload_lookup*`, `refresh_plan` | GPU-driven expert offload: device-side LRU over a slot pool, misses read over PCIe and installed while computing |
| `host_tier_*`, `host_post`, `host_wait`, `moe_layer_host` | experimental: misses computed by AVX-512 CPU workers through a mapped-memory mailbox |

All ops have fake (meta) implementations, so `torch.compile` + CUDA graphs capture them.

## Numbers (B200, Qwen3.8-Flash-Next 4-bit experts, 512 experts top-10)

Expert matvecs, gate_up + down per layer, random experts each call (from HBM, not L2):

| tokens | scalar | `qmv4_tc6` |
|---:|---:|---:|
| 1 | 30.4 us | 22.4 us |
| 3 | 69.3 us | 44.9 us |
| 4 | 85.1 us | 53.0 us |

End to end, `generate` with `torch.compile` (max-autotune, static cache), all experts resident: 110.3 -> 117.2 tok/s.

## Builds

`build/torch211-cu130-x86_64-linux` (torch 2.11, CUDA 13.0, `sm_100a` only) was built with `build_local.py`
(local nvcc + g++, `-march=sapphirerapids` for the host tier), not with kernel-builder. Other variants: build from
`csrc/` + `torch-ext/` with `build_local.py`, or with kernel-builder from `build.toml`.
