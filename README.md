# MoE inference on Apple Silicon (MPS): Inkling-2-Mini and Qwen3.8 notes

Hill-climb logs, harnesses and Metal kernels from making transformers' `generate` fast on an M3 Max (96 GB, ~290 GB/s measured)
for two MoE models. Everything was measured with control / candidate / control A/B runs (6 samples each, AC power, cooled GPU);
the ledgers hold every run.

## Inkling-2-Mini (`inkling2/`)
| metric | start | now | ceiling |
|---|---:|---:|---:|
| decode, opt-in 4-bit dense (native NVFP4 / MXFP8 experts) | 55.6 tok/s | 59.2 | 96 |
| decode, exact bf16 dense | 38.1 | 38.5 | 52 |
| prefill @512 tokens | 173 tok/s | 434 | ~1500 |
| prefill @2048 tokens | 111 tok/s | 499 | ~1500 |

- `inkling2/hillclimb/PLAN.md`: every experiment (E1-E13) with numbers, verdict and reason, plus the STATUS section.
- `inkling2/hillclimb/THEORY.md`: ceilings (bytes per token, bandwidth).
- `inkling2/hillclimb/ledger.jsonl`: one line per run. Harnesses: `hc_bench.py` (decode), `hc_prefill.py`, `hc_ablate.py`,
  `hc_prefill_ablate.py`, `hc_spec.py` / `hc_lookup.py` (speculation), `queue.sh` (serialized A/B with cooldowns).

Main finding: decode is GPU-bound at 16.5 ms/step. Host encode is 7.5 ms and overlaps; every hot kernel runs at 75-90% of its bandwidth
floor. The rest is serialization: torch's MPS stream encodes ~1200 small kernels per token into a serial compute encoder
(`MPSStream.mm`, `computeCommandEncoder` = `MTLDispatchTypeSerial`), so no two kernels overlap. Related upstream: pytorch/pytorch#206006
(host-signalled events, the host <-> GPU side of the same problem).

## Qwen3.8 experts offload (`qwen3.8-experts-offload/`)
Streaming MoE experts from SSD on MPS (huggingface/transformers#50997): `BENCH_LOG.md` holds the measurements, plus the synthetic
checkpoint builder, harness and quality / generate checks.

**2026-11-24:** [32 GB findings](qwen3.8-experts-offload/latest-2026-11-24/): final
pre-quantized checkpoint, transformers cache/read-ahead/lookahead results, the
reproducibility failure, the overlap/hit-rate ceiling (55 tok/s at 90% hit / 7.3 GB/s),
a 4-bit 96→32 GiB ladder with one identical output hash, and a 32 GB 50+ tok/s
2.5-bit build.

## Kernels (`kernels/`)
- `metal-moe-decode/`: the Metal kernels as published on the Hub (`ArthurZ/metal-moe-decode`): NVFP4 / MXFP8 expert decode
  (gate_up_act, down_route), grouped fp4 expert GEMMs for prefill, prefill flash attention with in-kernel relative bias / sinks,
  4-bit qmv4 (+ multi), router top-k, Inkling fused ops (attention inputs, message gate, attention residual, short conv).
- `experiments/`: kernel experiments with their benches, including the rejected ones (batched grid-synced message gates, wide GEMM tiles).
