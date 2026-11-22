# Inkling-2-Mini decode hill-climb (MPS, transformers generate, static cache, torch.compile)

Targets / constraints: c1 decode tok/s; keep quality (exact checkpoint numerics; the 4-bit dense path is an opt-in,
labelled lossy trade); context must stay stable past the 1024-token sliding window. Memory: 96 GB unified, model ~10 GB.

## Log (before this file; single runs, GPU contention noted, so treat small deltas as noise)
| step | tok/s | verdict |
|---|---:|---|
| PR #51050 eager, bf16-dequantized experts | (46 GiB) | baseline |
| native NVFP4/MXFP8 experts (#51062), exact dense | 29.6 | kept: 9.9 vs 46.2 GiB, logits equal |
| + Metal short convs (hub routing) | 29.9 | kept, ~noise |
| + fused decode attention | 30.2 | kept |
| + fused block attention residual | 32.9 | kept |
| + fused message gate | 34.2 | kept (exact config best) |
| opt-in 4-bit dense + lm_head (bench hack) | 48.3 | kept as labelled lossy option |
| grouped 6 attention projections (one qmv4) | 48.0 | REJECTED: inside noise, dispatches were not the cost |
| inductor asserts off | 28.4 vs 29.9 | REJECTED: host time overlaps GPU here |
| cpp_wrapper | n/a | REJECTED: 437k-line C++ never finished compiling (~50 min) |
| plan path: stored experts + {"*": MetalConfig(4)} | 32.1 -> 48.6 | fixed MetalConfig.is_compileable (compile was silently off) |
| + optimized fp4 expert kernels | (in 48.6) | kept |
| + fused router | 37.8 (GPU 88% busy) | kept (kernel 33 vs 62 us), number void |
| + faster attn_res / 1-dispatch message gate / fused short-conv step | 55.0 | kept |
| + fused attention inputs | 55.4 (GPU 99% busy) | kept, needs a clean A/B |
| 900-token generation | 56.3 | stable |
| DSpark speculation (7 drafts) | 18.2 vs 33.8 greedy | REJECTED: acc 2.4 incl. bonus, T=8 verify ~4x T=1 |
| >1024-token static generation | crash | FIXED: hybrid_sliding mask mapping (bug since Inkling #47347) |

## Experiments (from here: control / candidate / control, >= 6 samples, medians, AC + idle GPU)

### 2026-11-22 baselines (AC, GPU idle, 6 samples x 128 tokens, medians)
| run | tok/s |
|---|---:|
| base-4bit | 55.54 |
| base-exact | 38.13 |
| base-4bit | 55.75 |
Run-to-run noise ~0.4%. Exact = 73% of its 52 tok/s ceiling, 4-bit = 59% of 95.
Also fixed: safetensors MPS dlpack OnceLock/GIL deadlock (upstream since #767) that hung threaded exact loads.

### E1 HC_GROUP: q / gate_attn / relative-states / sink projections as one matmul (control-candidate-control, both paths)
- attempt 1 VOID: base-4bit 55.70, grp-4bit 27.00, base-4bit 24.53 (control halved too: environment changed mid-queue), then
  the queue died silently in run 4 (exact control, after compile). No thermal/power warning recorded. Rerun.
- attempt 2: base-4bit 31.88 | grp-4bit 41.09 | base-4bit 41.48 ; base-exact 28.65 | grp-exact 27.86 | base-exact 25.41.
  Controls far below the morning baselines (55.6 / 38.1) and swinging 30%: environment degraded (no thermal/power flag;
  the GPU busy check before each run passed). VERDICT: REJECTED (does not beat both neighbouring controls on either
  path; also rejected once before). The rest of the hill-climb waits for a stable machine (baseline A/A within ~2%).

### Ablation of the current best (4-bit plan, stable machine: A/A 55.82 / 55.63)
| removed | ms/token saved (of 17.74) |
|---|---:|
| experts | 9.46 |
| 4-bit dense matvecs (~370 qmv4 + lm_head) | 7.37 (floor ~3.2: biggest gap) |
| message gate | 1.46 |
| attention | 0.93 |
| router | 0.79 |
| attention residual | 0.67 |
| short conv | 0.32 |
Next: E1 rerun on a stable machine (grouping is the lever for the latency-bound thin matvecs).
- E1 rerun 1: base 55.42 | grp 57.63 | base 34.62 (VOID: samples decay 38.0 -> 31.5 within the run = thermal throttling
  after ~2 back-to-back runs). Harness now voids runs whose last sample is >5% below the first; queue cools 180 s between runs.
- E1 rerun 2 (180 s cooldown before each run): base 50.89 | grp 57.49 | base 55.71, no run throttled, outputs coherent.
  VERDICT: ACCEPTED (+3.2% over the better control; repeats the earlier clean 57.63). Grouping the 4 projections
  that read hidden_states into one qmv4 cuts the latency-bound thin matvecs (sinks 24 rows, relative states 384).
  Next: productize (fuse at load for MetalLinear), then E2 = the same for the exact bf16 path.
- prod-4bit (fusion at load in the Metal quantizer): 57.50 (matches the accepted 57.49). Best = 57.5 tok/s.

### E2 HC_GROUP on the exact bf16 path (concatenated weights, one F.linear)
- base-exact 37.93 | grp-exact 38.55 | base-exact 37.91: ACCEPTED (+1.6%, noise 0.4%). Exact path has no quantizer to fuse at load and fusing renames state_dict keys (saving): kept as an opt-in helper for now.

### E3 k/v sibling group (members with another input run their own rows)
- q-group 57.57 | qkv-group 57.46 | q-group 57.53: REJECTED (inside noise). k/v group reverted; the own-rows slice for members called on another input is kept (correct, free).

## STATUS (2026-11-22)
| metric | PR #51050 baseline | now | ceiling | harness |
|---|---:|---:|---:|---|
| c1 decode, exact (bf16 dense, native fp4 experts) | ~15 (eager, 46 GiB) | 38.6 (opt-in grouping) / 38.1 | 52 | hc_bench, median of 6 x 128 tok, AC, cooled |
| c1 decode, opt-in 4-bit dense | n/a | 59.2 (E8) | 95 | same |
| memory | 46.2 GiB | 9.9 GiB | | |

What limits further gains (ablation of the 17.74 ms 4-bit step): experts 9.46 ms (floor ~6.3), 4-bit dense matvecs ~6.8 ms
(floor ~3.2; thin shapes are latency-bound at ~10 us per dispatch), message gate 1.46, attention 0.93, router 0.79,
attention residual 0.67, short conv 0.32. Speculation (DSpark) loses on this target: acceptance 2.4 incl. bonus and an
8-token verify reads the union of experts. 150 tok/s is above every ceiling of this checkpoint on this machine.

Remaining levers, ranked (estimated gain / cost):
1. experts per-evaluation fixed cost + decomposition (~1-3 ms / kernel work; agent running)
2. o_proj + gate epilogue fused into one matvec; k/v projections fused with their conv tails (~0.5-1 ms / kernel work)
3. message gate down-projection fused into the attention residual that produces its input (~0.4 ms / signature change)
4. exact path: grouping productized (save-safe fusion) (+1.6% measured)
Thermal: back-to-back runs throttle after ~2 runs; the harness voids runs whose samples decay >5% and the queue cools 180 s.

### E4 cast-free expert kernels (int64 ids / bf16 weights read in-kernel)
casts 57.92 → nocasts 58.25 → casts 58.43 tok/s (4-bit dense, 6 samples each, none void).
Verdict: REJECTED: the candidate does not beat the second control. The ~0.3 ms/token the microbench predicted is inside run noise (~1%).
Reverted to pin 8d84318 and restored the casts. Hub rev 51ab38ad stays published (unused).

### E5 small-kernel bundle: cast-free experts + router v2 (split rows, register selection), hub rev a2400044
Router microbench (GPU events, real shape, DRAM): 21.4 -> 16.0 us/call; expected ~0.55 ms/token with E4 (~3%).
casts 57.96 → bundle 58.76 → casts 57.86 tok/s (6 samples each, none void; bundle min 58.52 > control max 58.29). Greedy text identical.
Verdict: ACCEPTED (+1.4%). The pin is now a2400044 and the casts are dropped.
Findings: a dispatch under compile costs ~1.1 us (400 chained kernels = 498 us), so the 1218 dispatches/token
(761 custom + 457 inductor) are only ~1.4 ms. The rest of the non-expert gap is the fixed latency of small dependent kernels
(router selection ~5 us after its last-threadgroup sync, message gate's two grid syncs ~15 us). Experts run at ~80% of 290 GB/s.

### E6 prefill: grouped fp4 expert GEMMs (T >= 64), hub rev e00b93b7
Baseline prefill (4-bit dense, compiled decode, eager prefill): 173 tok/s @512, 111 @2048. Experts were 3.57 of 4.64 s @512 (matvec per pair).
GEMM (pairs sorted by expert, bf16 exact tiles, fp32 MMA): 3.1-4.7x per layer; prefill 321 tok/s @512, 213 @2048 (+86% / +92%).
Locals reference (M4 Pro): plain decode 37 tok/s, prose 36; 63-76 only with drafts on copy-heavy tool output.

### E7 qmv4_multi: q-group / k / v matvecs in one dispatch (hub rev 825f8356c52b34fbdddb1db604b91bef333a61f0)
Microbench: bit-exact, 90.3 -> 69.2 us per layer pass (eager). qmv4 census: 6.56 ms vs 3.40 floor; k/v 1024x2304 is latency-bound (12.6 us vs 4.6).
prev 57.75 → qkv-multi 57.90 → prev 57.00 tok/s (none void; greedy text identical). The compiled graph goes from 209 qmv4 to 65 qmv4 + 48 qmv4_multi.
Verdict: KEPT (bit-exact, beats both controls) but the gain is marginal (+0.3% vs the nearer control). The 21 us/layer the eager microbench saved
was host overhead that compile already hides, consistent with ~1 us per dispatch under compile.
Lesson: dispatch merging is exhausted. What's left is the efficiency inside the kernels: experts gate_up 83% / down_sum 71% of 290 GB/s,
fused-q matvec 70%, o_proj 71%, k/v 37% (latency-bound shape).

### E8 down_route: one threadgroup per (pair, 32-row block), fixed-order route sum (hub rev d6dfa9bcbff45fea40afc376a4d298735b3a1b17)
Microbench (16 distinct routings, DRAM): nvfp4 down 70.6 -> 55.9 us, mxfp8 97.6 -> 90.4 us; rel err vs down_sum within bf16 (2-3e-3).
prev 58.41 → down-route 59.24 → prev 58.48 tok/s (none void; greedy text identical; candidate min 58.97 vs control max 58.91).
Verdict: ACCEPTED (+1.4%). Best so far is 59.2 tok/s, 4-bit dense.

### E9 three message gates (+ norms) in one dispatch: REJECTED before A/B
Kernel: 3 x 16 threadgroups of 1024 threads, each gate grid-syncing its own 16. Correct (sliding 3e-3 = bf16 of the norm; full exact),
but 19 ms per call: one gate's 16 threadgroups are not co-resident beside the others, so its spin-wait grid sync runs to MG_MAX_SPIN.
Grid-synced kernels can't be batched past co-residency. A sequential in-kernel loop over the gates would only save the launches (~1 us each). Reverted.

### E10 prefill flash attention: bias / window mask / sinks in-kernel (hub rev ef3008e4)
Prefill ablation before (GEMM in place): attention 159 ms of 1548 @512 and 4174 of 8618 @2048 (dense [H, T, L] bias, mask and softmax temporaries).
Kernel microbench vs eager: 2048 sliding 138 -> 8.1 ms, full 204 -> 11.1 ms; rel err 7-8e-3 (P rounded to bf16 against the running max).
Same-session prefill: 290 -> 356 tok/s @512 (+23%), 193 -> 399 tok/s @2048 (+107%).
Caveat: within bf16 tolerance, not bitwise. A 941-token prompt's greedy continuation diverges at token ~13, the same kind of drift
as switching eager for sdpa.

### E11 prefill GEMM: 64-row tiles once experts average 64+ rows; fp32 output tile aliases the staging buffers (hub rev bb89132a)
Microbench nvfp4 2048: 75 -> 63.5 ms/layer (4.1 TFLOPS), 4096: 4.5-4.9 TFLOPS. Same session vs ef3008e4: prefill 385 -> 434 tok/s @512,
443 -> 499 @2048; the 941-token greedy continuation is identical.
Prefill since the start of the hill-climb: 173 -> 434 tok/s @512 (2.5x), 111 -> 499 @2048 (4.5x).

### E12 exact path: q / k / v bf16 matvecs in one dispatch (bmv_multi, hub rev d5651f5727376058111301857024adbf3e20354f)
Microbench: torch bf16 linear 1024x2304 43.1 us (118 GB/s) -> bmv 28.5 us; bmv_multi bit-exact vs F.linear.
exact-prev 38.49 → exact-bmv 38.25 (the closing control deadlocked in safetensors' MPS dlpack GIL bug, killed). The compiled graph goes
from 353 mm to 209 mm + 48 bmv_multi, so the path does run.
Verdict: REJECTED. The eager microbench (43 -> 28.5 us) did not carry into the compiled graph, where torch's mm is not as slow. Pin reverted to bb89132a.
Lesson: microbench inside a compiled graph (or in-context ablation), never eager, before building dispatch / small-shape levers.


## STATUS (2026-11-22, after E12)
| metric | start of hill-climb | now | ceiling (290 GB/s / ~14 TFLOPS) |
|---|---:|---:|---:|
| decode, opt-in 4-bit dense | 55.6 | **59.2** | 96 (3.02 GB/token) |
| decode, exact bf16 dense | 38.1 | 38.5 | 52 |
| prefill @512 | 173 | **434** | ~1500 |
| prefill @2048 | 111 | **499** | ~1500 |

Where decode stands (measured): GPU-bound at 16.5 ms/step. Host encode is 7.5 ms and hides behind the previous step; the generate
loop's host gap is ~0.9 ms. ~1200 dispatches/token at ~1.1 us launch cost each. Every hot kernel is at 75-90% of its practical (272 GB/s) floor:
experts gate_up 89%, down_route ~85%, fused-q matvec 84%, o_proj 76%, lm_head 94%. What's left is serialization: torch's MPS stream
encodes every kernel serially (implicit barrier between dispatches), so ~1200 dependent small kernels each pay ramp-up and tail with no
overlap. Batching across grid-synced kernels is capped by co-residency (~16 threadgroups of 1024 threads: E9, MG_TGS=24 both spin out).
Not reachable on this stack without lossy trades: 100 tok/s is above the 96 tok/s ceiling, and even 80+ would need kernels running
concurrently (an MPS backend change to concurrent dispatch with explicit barriers) or a persistent per-layer megakernel.

Ranked remaining levers:
1. Prefill expert GEMM (5-6.6 TFLOPS gate_up): double-buffered k loop / larger tiles at 2048+ (prefill +10-20%).
2. Prefill "other" (~250 ms @512): router / conv / norm prefill paths as kernels.
3. Decode: o_proj gate epilogue fused into attention (-48 kernels, ~0.2 ms), KV write fused into attention_inputs (~0.3 ms).
4. PyTorch MPS: concurrent dispatch encoder (upstream work), the only lever with multi-ms decode headroom.

### E13 prompt-lookup speculation, compiled verify (copy-edit workload, reasoning effort 0): REJECTED
Fixes to get it running: prefill_attention as a torch library op (hub rev 36e2b080, also makes compiled prefill / verify graphs
possible), drafts padded to K (two verify shapes), prefill kept eager, 16 tokens of cache headroom, dynamo cache limit raised.
Result: greedy 48.5 tok/s vs lookup-k4 22.4 tok/s; 1.69 accepted drafts per verify; ~120 ms per verify step vs ~20 ms per decode step.
Blockers: (1) Inkling still writes a thinking preamble at reasoning effort 0, so lookup rarely matches; (2) the verify path graph-breaks
in every layer and the conv states change shape while the past is recorded (4 -> 4 + K + 1), so it can't use the tuned decode graph.
Locals' 2x needs near-verbatim tool output plus an engine-level verify; neither applies to transformers generate here.
