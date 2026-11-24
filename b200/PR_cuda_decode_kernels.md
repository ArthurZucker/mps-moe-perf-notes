Stacked on #50997 (`stream-experts`): compiled decode (`torch.compile` + static cache + CUDA graphs) for Qwen3.8-Flash-Next with every expert resident on CUDA, with the experts running on a Hub kernel.

## What was missing

1. **Compiled decode crashed.** With a `StaticCache`, the indexer position history grew by concatenation on the cache layer: its shape changed every step (a recompile per token), and under CUDA graphs the stored tensor was a graph output that the next replay overwrote (`RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`). `StaticIndexedLayer.update_indexer_positions` now writes positions in place into a `[dims, batch, max_cache_len]` buffer (static address) and returns the buffer followed by the current positions; `DynamicIndexedLayer` keeps concatenating. The model calls `indexer_cache.update_indexer_positions(position_ids)`. Regression test in `tests/utils/test_cache_utils.py` (fails on `stream-experts`, passes here).
2. **`MetalExperts` on CUDA ran the torch path** (dequantize the routed experts, then matmul). For 1-16 tokens they now run on the pinned Hub kernel [`ArthurZ/moe-affine`](https://huggingface.co/kernels/ArthurZ/moe-affine) (`bb5c0529`, sm_100 build): a tensor-core expert matvec (`qmv4_tc6`: per block, one `cp.async` batch stages a row tile of every routed expert; codes are dequantized as bf16 `128 + c`; `mma.m16n8k16`), with scalar kernels as fallback. Down is fused with the routing-weighted sum. The ops have fake implementations, so compile and CUDA graphs capture them. Loading goes through a new `cuda_kernel()` (pinned repo and revision, like `metal_kernel()`), resolved on the first eager call; all dispatch lives in `integrations/cuda_affine_moe.py`.

## Numbers

B200, `ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit` (all experts resident), 4 prompts × 128 greedy tokens, `generate(..., cache_implementation="static", compile_config=CompileConfig(mode="max-autotune"))`:

| | full (eager) | full + compile |
|---|---:|---:|
| `stream-experts` | 13.0 tok/s | crash |
| + static indexer positions | 13.1 | 43.0 |
| + `moe-affine` experts (this PR) | 13.9 | **117.1** (8.54 ms/token) |

Expert matvecs alone (gate_up + down per layer, random experts per call, CUDA graph): 1 token 30.4 → 22.4 µs, 3 tokens 69.3 → 44.9 µs, with the scalar kernels as the "before".

Greedy compiled output matches eager for the first 84 tokens of prompt 1, then diverges at a bf16 near-tie (the plain torch compiled path diverges at token 95).

## Not in this PR

- Sibling-projection fusion + small-row GEMV under `use_kernels` (`cuda-decode-linears-use-kernels`): +12% eager, nothing under compile.
- MTP speculative decoding (K=2): ~134 tok/s resident on the same machine, done outside the library.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
