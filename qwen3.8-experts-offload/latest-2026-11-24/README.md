# Qwen3.8-Flash-Next: 32 GB findings (2026-11-24)

This directory is the 2026-11-24 follow-up to `../BENCH_LOG.md`. It covers:

1. **Final checkpoint**: the always-on layers stored pre-quantized, so loading does no
   `quantize_dense` work. Published as
   `ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit-final`.
2. **transformers stream-experts**: cache sweep, read-ahead pinning, lookahead sweep,
   CPU/GPU split, and the correctness/reproducibility failures found on the way.
3. **The actual 32 GB ceiling**: why 32 GB is not physically impossible, and what has to
   be true to reach 50 tok/s (overlap + hit rate + miss bandwidth). See `CEILING.md`.
4. **A 4-bit reference ladder** from 96 GB to 32 GB with identical output hashes, on the
   MLX/JITEngine reference (not transformers).
5. **A 32 GB, 50+ tok/s point** on a 2.5-bit llama.cpp GGUF (not transformers; quality
   caveat). This is the only configuration measured at 50+ tok/s with a 32 GB footprint.

## Headline numbers

### transformers stream-experts (4-bit final checkpoint)

| expert cache | events | decode | sustained | correctness |
|---:|---|---:|---:|---|
| 6 GiB | off | 4.71 | 5.35 | 128/128 |
| 16 GiB | off | 8.56 | 9.10 | 128/128 |
| 16 GiB | on + pin-ahead | 14.41 | 18.86 | **5/128** |
| 16 GiB | on + pin-ahead, short | 14.57 | 15.15 | text check only |

Same prompt, same cache, two runs produce **different token hashes** even eager, with
hub kernels disabled and deterministic top-k. The MoE quant kernels themselves are
deterministic in isolation (`gather_qmm`, `qmv4`). The nondeterminism is elsewhere in the
Metal path and is not yet fixed. Details in `RESULTS.md`.

### 4-bit reference ladder (JITEngine MLX, not transformers)

512-token interactive benchmark, 4-bit checkpoint, same task:

| budget | expert slots | decode tok/s | aggregate tok/s | footprint | text hash |
|---:|---:|---:|---:|---:|---|
| 32 GiB | 9,122 | 30.88 | 20.00 | 31.15 GB | `07a676778e3a6720` |
| 40 GiB | 13,083 | 45.01 | 25.84 | 42.10 GB | `07a676778e3a6720` |
| 48 GiB | 16,190 | 59.59 | 31.43 | 50.84 GB | `07a676778e3a6720` |
| 64 GiB | 22,404 | 69.80 | 48.43 | 67.88 GB | `07a676778e3a6720` |
| 96 GiB | 24,576 | 50.22 | 38.64 | 73.00 GB | `07a676778e3a6720` |

The 96 GiB row is slower because the engine turned MTP off there (the main pool held all
24,576 experts, leaving no extension for the draft head). 64 GiB is the best of the ladder.
All five budgets produce **byte-identical** output (`07a676778e3a6720`), so the 4-bit
quality is independent of the memory budget; only speed changes.

### 32 GB, 50+ tok/s (2.5-bit, non-transformers)

`webmp3/Sakura-Qwen3.8-Flash-Next-Swift-2.5bit-352E-32GB-GGUF` on the patched
`danielhanchen/llama.cpp` `qwen4exp/mtp` build:

- 56.4 / 58.3 / 59.2 / 62.2 tok/s decode over unique prompts, 512 tokens
- 30 GB physical footprint, flat swap
- MTP on, 2 drafts, `-ngl 95`, 2048 context
- 2.5-bit, 352/512 experts: a quality trade, not the 4-bit path.

## Files

- `CEILING.md`: the overlap/hit-rate/SSD ceiling, including the 55 tok/s point.
- `RESULTS.md`: full tables, hashes, nondeterminism experiments, Sakura details.
- `PATCHES.md`: the transformers patches in this directory and how to apply them.
- `patches/`: 0001 read-ahead pinning + lookahead env; 0002 deterministic top-k.
- `tools/`: final-checkpoint builder, final 32 GB harness, short benchmark.
- `results/`: raw benchmark JSON and logs (paths scrubbed).

## Status

- Final checkpoint: done and published; 0 on-the-fly quantization.
- transformers 4-bit 32 GB: best correct run 8.56/9.10; fastest run 14.41/18.86 but 5/128.
- 32 GB ceiling with overlap: ~55 tok/s at 90% hit and 7.3 GB/s; see `CEILING.md`.
- 32 GB 50+ tok/s: only the 2.5-bit non-transformers build.
