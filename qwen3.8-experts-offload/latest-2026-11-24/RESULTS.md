# Results and raw evidence

## 1. Final checkpoint

`ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit-final` (`aefb54d`) stores the 725
always-on `nn.Linear`s pre-quantized (Metal 4-bit, group 32, clipping search). Only the
52 files holding dense layers were rewritten; expert and n-gram files are APFS clones of
the parent. Load prints:

```text
quantized 0 nn.Linear: 4g32c, fused 192 sibling groups
on-the-fly quantization calls: 0
```

Tools: `tools/make_final_qwen38_checkpoint.py`, `tools/run_qwen38_metal_final.sh`.

## 2. transformers cache sweep (events off)

| cache | decode tok/s | sustained | peak footprint | peak MPS driver |
|---:|---:|---:|---:|---:|
| 6 GiB | 6.66 | 6.70 | 12.87 GiB | 10.78 GiB |
| 8 GiB | 6.95 | 7.06 | 15.16 GiB | 12.47 GiB |
| 10 GiB | 7.43 | 7.43 | 17.43 GiB | 15.35 GiB |
| 12 GiB | 7.80 | 7.92 | 19.14 GiB | 17.04 GiB |
| 14 GiB | 8.12 | 8.14 | 21.10 GiB | 19.03 GiB |

Host sync is flat at ~37 ms/token; cache size only recovers part of it.

## 3. transformers events-on + pin-ahead

Patch `0001`: read-ahead may not evict any resident expert of the layer it is
prefetching; add `TRANSFORMERS_EXPERTS_OFFLOAD_LOOKAHEAD` and
`TRANSFORMERS_EXPERTS_OFFLOAD_PIN_AHEAD`.

| cache | decode | sustained | peak footprint |
|---:|---:|---:|---:|
| 6 GiB | 8.02 | 8.20 | 12.07 GiB |
| 10 GiB | 9.84 | 10.01 | 16.64 GiB |
| 12 GiB | 10.99 | 10.91 | 18.32 GiB |
| 14 GiB | 11.77 | 11.63 | 20.33 GiB |

Short 512-token runs completed without the `KeyError`. Full `verify_generate.py` at
16 GiB then produced **5/128** compiled == eager, so the patch is not a correctness fix.

## 4. Lookahead sweep (12 GiB, events on, pin-ahead)

| lookahead | decode | sustained | ahead reads |
|---:|---:|---:|---:|
| 0 | 12.01 | 11.96 | 0 |
| 1 | 11.06 | 11.14 | 65,524 |
| 2 | 10.92 | 10.77 | 65,890 |

Lookahead 0 wins on the real model; the guesses covered 0 actual misses
(`covered=0`) and added host work.

## 5. Reproducibility failure

Same prompt, same 16 GiB cache, two separate processes, eager, hub kernels disabled,
deterministic experts and deterministic top-k:

| run | token hash |
|---|---|
| eager run 1 | `1d2dccbaed3c7374` |
| eager run 2 | `d04657441c68cd5f` |
| no-hub run 1 | `59d73f4047ec0b44` |
| no-hub run 2 | `a25123907093de52` |
| deterministic-topk run 1 | `67981c5e74161129` |
| deterministic-topk run 2 | `353640103a1fe075` |

At 6 GiB: `34e6260a7ab9862c` vs `ba3f555a08f56cd2`; at 16 GiB (compiled):
`3945249474294e97` vs another. The generated text is coherent but the greedy tokens
diverge after a few steps.

What is **not** the cause:

- `gather_qmm`: 10 repeated calls on the same inputs are bit-identical.
- `qmv4`: 10 repeated calls are bit-identical.
- router top-k order: stable sort + id-ordered top-k did not fix it.
- compiled graph: eager has the same divergence.
- hub GDN/conv kernels: disabling them did not fix it.
- fused expert decode: disabling `_fused_decode` did not fix it.

Still open: the remaining nondeterminism is in the Qwen4 Metal path (QSA/attention,
GDN fallback, or another reduction). Until this is fixed, no 96→32 GB output ladder can
be reproducible on the transformers path.


## 5b. Measured 32 GB decomposition (JITEngine `/stats`)

After the 512-token benchmark at 29.8 GiB budget, `/stats` reports:

- decode hit rate: **96.77%** (622,438 hits / 787,200 lookups)
- misses: 20,769; miss bytes: **57.42 GB** over 1,544 tokens = **37.2 MB/token**
- stall: 13.15 s / 1,544 tokens = **8.52 ms/token**
- decode wall: 33.3 ms/token (30.0 tok/s in this sample)
- non-stall (host + GPU): 33.3 - 8.5 = **24.8 ms/token**
- MTP: 816 verifies, 716 accepted, 1.89 tokens/round

The hit rate is already 96.8%, not 80-90%. The miss path is only ~37 MB/token. With
perfect overlap the ceiling under the current non-stall cost is `1000/24.8 = 40.3 tok/s`,
not 55. The 48 GiB run's 59.6 tok/s (16.8 ms/token) shows that the 32 GiB non-stall cost
is about 8 ms/token higher: cache bookkeeping / host work / MTP verify, not miss bandwidth.
To reach 50 at 32 GB, the non-stall path has to drop below ~20 ms/token, or the extra
host work has to move off the critical path.

## 6. 4-bit reference ladder (JITEngine MLX, not transformers)

512-token interactive decode, same task, `text_sha` from the benchmark JSON:

| budget | slots | decode | aggregate | footprint | text_sha |
|---:|---:|---:|---:|---:|---|
| 32 GiB | 9,122 | 30.88 | 20.00 | 31.15 GB | `07a676778e3a6720` |
| 40 GiB | 13,083 | 45.01 | 25.84 | 42.10 GB | `07a676778e3a6720` |
| 48 GiB | 16,190 | 59.59 | 31.43 | 50.84 GB | `07a676778e3a6720` |
| 64 GiB | 22,404 | 69.80 | 48.43 | 67.88 GB | `07a676778e3a6720` |
| 96 GiB | 24,576 | 50.22 | 38.64 | 73.00 GB | `07a676778e3a6720` |

All hashes identical: 4-bit quality is memory-budget independent. The 96 GiB row turns
MTP off (all experts in the main pool leaves no extension), hence lower than 64 GiB.

## 7. 32 GB, 50+ tok/s (2.5-bit GGUF, not transformers)

`webmp3/Sakura-Qwen3.8-Flash-Next-Swift-2.5bit-352E-32GB-GGUF` + shared MTP Q4_K_M on
`danielhanchen/llama.cpp` `qwen4exp/mtp` @ `6fcaa16`:

| run | decode tok/s | wall | usage |
|---|---:|---:|---:|
| mem1 | 62.18 | 8.54 s | 512 |
| mem2 | 59.71 | 8.88 s | 512 |
| test1 | 56.41 | 9.40 s | 512 |
| test2 | 58.28 | 9.09 s | 512 |
| test3 | 59.21 | 8.96 s | 512 |

`footprint`: 30 GB physical, peak 30 GB; swap flat. 2.5-bit, 352/512 experts, MTP on,
`-ngl 95`, 2048 context, `-b 256 -ub 256`. This is a quality trade and not the
transformers path. It is the only 50+ tok/s result at a 32 GB footprint.

## 8. Kernel determinism check

`results/kernel_determinism.log`: `gather_qmm` and `qmv4` repeated calls are byte-identical
on real packed expert weights. The nondeterminism is not in those two kernels.

## 9. Raw files

- `results/jitengine_bench{32_mtp,40,48,64,96}.json`
- `results/transformers_repro.log`, `eager_repro.log`, `dettopk_repro.log`
- `results/kernel_determinism.log`
- `results/sakura_bench_unique.log`
- `results/memtrace_final.tsv`, `memtrace_e1_6g.tsv`, `memtrace_e1_16g.tsv` (where present)
