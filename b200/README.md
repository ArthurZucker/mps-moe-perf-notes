# Qwen3.8-Flash-Next on one B200: decode with 25% of the experts on the GPU

Model: `ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit` (48 layers, 512 experts top-10, affine 4-bit g64 experts =
2.765 MB each, 64.6 GiB of experts; hyper-connections x4, GDN + indexed attention). Host: 8x B200, 2x Xeon 8570
(56 cores/socket, 1 TB DDR5 per socket), PCIe Gen5 (51-55 GB/s measured H2D). Bench: `bench.py`, 4 prompts x 128 greedy
tokens, `torch.compile` max-autotune + static cache + CUDA graphs. Ledger with every experiment: [`PLAN.md`](PLAN.md);
ceilings: [`THEORY.md`](THEORY.md).

## Where we stand

| config | decode tok/s | GPU mem steady / peak | tokens vs full |
|---|---:|---:|---|
| full + compile, torch 2.15 / kernels 0.19 | **119.6** | ~120 / 124.8 GiB | ref |
| full + compile, torch 2.11 (tc6 experts) | 118.6 | ~120 / 124.8 GiB | ref |
| full + compile, MTP K=2 speculative | ~134 | ~125 GiB | bf16 near-ties |
| **0.25: CPU computes the misses** (host tier v2, best run) | **83.9** | 74 / 120.3 GiB | 128/128 |
| 0.25: host tier v2, other runs | 73-82 | 74 / 120.3 GiB | 128/128 |
| 0.25: GPU-only streaming, scalar kernels | 67.9 | 74 / 120.3 GiB | 128/128 |
| 0.25: GPU-only streaming, tc6 kernel | 54.9 | 74 / 120.3 GiB | 110/128 |
| start of this round: 0.25 + compile | ~70 | | |

Experts at 0.25: 17 GB on the GPU instead of 68 GB. The 120 GiB peak is the load / prefill staging, not decode.
Goal (not reached yet): misses cost nothing, i.e. 0.25 at the full-residency rate.

Where a 0.25 token goes (GPU timestamps in the graph, `stamps2.py`, ms/token):

| | full | 0.25 host tier |
|---|---:|---:|
| layers 1..47 | 6.65 | 9.85 |
| layer 0 MoE | 0.05 | 0.21 |
| layer 47 end -> next token's layer 0 (lm head, sampling, Python loop, launch) | 2.07 | 2.25-2.65 |

The remaining ~3.5 ms is the GPU waiting for the CPU at the join: a layer's misses (~2.5 experts, ~7 MB) are read at
one socket's DRAM bandwidth (~140-150 GB/s), ~45 us against a ~25 us resident window.

## What it took (wins, each A/B'd)

1. **Expert kernel `qmv4_tc6`** (`kernel-src/moe-affine`, on the Hub as `ArthurZ/moe-affine`): per block, one cp.async
   batch stages a 16-row tile of every routed expert + scales + activations; codes -> bf16 `128+c` by one `lop3`, then
   `mma.m16n8k16`. gate_up+down per layer: 30.4 -> 22.4 us (1 token), 69 -> 45 us (3 tokens). Full: 110 -> 117 tok/s.
   Ablations (kbench/sa): scattered per-lane weight/scale loads were the cost, not the math; ldg/cp.async/TMA within 10%.
2. **Host compute tier v2** (`csrc/host_tier2.cpp`): CPU workers compute the missed experts from the pinned banks.
   - codes as bf16 pairs `((w >> 4k) & 0x000F000F) | 0x43004300` (nibbles k, k+4) + `vdpbf16ps`, 4 rows interleaved:
     13.1 GB/s/core (8.9 before); activation sent in pair layout + group sums by the GPU.
   - dissemination barrier instead of a 55-thread `fetch_add`; gate+act merged (one barrier before down).
   - 1 miss: 65-75 -> 33 us per layer serial.
3. **Host-tier GPU side off the critical path**: post / poll / result copy on a side stream, joined before the add;
   refresh + next-layer guess on a background stream joined once per token. With oracle hints: +3 us/layer at 1 miss.
4. **Pool refresh by the copy engine**: GPU plans LRU victims (slots of layers <= current, unused this token), the key
   stays pending until a CPU installer thread's `cudaMemcpyBatchAsync` lands (`cuStreamWriteValue64` publishes it), the
   GPU commits only if the slot owner is unchanged. Byte-exact test. 72.5 -> 76.2 tok/s.
5. **Main Python thread off the spinning workers' socket**: 73 -> 82-84 tok/s (spinners eat the socket's turbo budget).
6. **No accelerate hooks after the store takes the experts**: 1302 modules kept a dispatch hook from the cpu-offload
   `device_map`, ~2 ms/token of host time (recursive input walk incl. the cache). Gap between tokens 4.2 -> 2.2 ms.
7. **Fixes found by checking instead of assuming**: tc6 flag read emptied banks (never on in the store); post/lookup race
   (a concurrent commit could drop a route: the CPU now takes exactly the routes the lookup gave GPU weight 0); NaN
   logits -> hint index -1 -> CPU segfault (guards both sides); undeclared side-stream state (declared mutations were
   functionalized into a racing copy-back: 41/128 tokens -> 128/128).

## Explored and rejected (measured)

| idea | result |
|---|---|
| tc6 for streamed misses (inline PCIe) | 54.9 vs 67.9 tok/s scalar: one tile in flight per block |
| `umwait` instead of spinning | +7 us per call wakeup; 74.9 tok/s |
| dual-socket workers with a NUMA replica of the banks | layers 1..47 9.85 -> 12.35 ms (cross-socket barrier) |
| wider hints (top-24, 12 staged) | +0.9 ms (staging competes with real misses for DRAM) |
| v3: lookup+post fused on the main stream, in-stream join | 10.3 vs 9.85 ms (post on the critical path) |
| cache policies (SLRU, ARC, LRU-2, decayed LFU +/- admission) | 63.6-74.6 misses/token vs LRU 64.6; Belady 34.7, per-prompt static oracle 39.8 |
| copy-engine contention as the cause of the gap between tokens | refuted (worse without refresh) |
| installer driver calls as the cause | refuted (batched installer: same gap) |
| torch 2.15 / kernels 0.19 | 119.6 vs 118.6 full (built and tested; C++20 needed, nvcc pinned to 13.2) |

Profiler caveat: `prof_decode.py` warms up on the same prompt, so the LRU pool is warm: it reported gpu-nola at
103.8 tok/s where the bench (4 distinct prompts) gives 54.9. Bench + in-graph GPU timestamps are the references.

## Next

- Hints from the streams entering a layer (its MLP-side hyper-connection mix, before attention): 89% of misses covered
  at top-16 vs 70% for the current guess (`miss_dist2.py`); fused kernel `hc_mix_guess` written and tested (3.7e-3 vs
  the module), being measured in the model.
- Upstream: accelerate-hook removal + static indexer positions + `moe-affine` dispatch (transformers draft branch
  `cuda-decode-kernels`), safetensors PR 996 (prefetch into slots), torch (`cudagraph` partition / custom-op stream
  notes).

## Layout

`bench.py` (configs), `stamps2.py` (in-graph timing), `prof_decode.py` (profiler), `miss_dist*.py`, `sim_*.py`
(cache policies), `kbench/` (kernel microbenches, CPU dot bench, host handoff harnesses, tests), `kernel-src/moe-affine`
(kernel sources + `build_local.py`), `mtp/` (MTP speculative engine), `logs/` (every run, guarded by `q.sh`).
