# Hill-climb: Qwen3.8 ours:0.25+compile decode on B200 -> target 300 tok/s

## Baseline (2026-11-22, PR #50997 head + static-cache indexer fix)
full eager 13.6 | full+compile 38.2 | ours:0.25 1.9 | ours:0.25+compile 3.8 tok/s (bench_experts_offload.py, 4 prompts x 128)

## Experiments (prof_decode.py: compiled decode median ms/token, full residency, 64 tokens)
- E1 experts: Triton gather_qmv (25 ms/token, ~70 GB/s, 8x redundant word loads) -> CUDA gather_qmv4 + down_route4
  (warp per row, uint4 loads, routing sum fused). 26.2 -> 11.05 ms (38 -> 90.5 tok/s). ACCEPTED. Greedy text identical.
  Packaged as HF kernel repo kernels/moe-affine (build/torch211-cu130-x86_64-linux, sm_100a), loaded via lazy_load_kernel.
- E2 dense: sibling fusion (192 groups) + GEMV kernel for N<1024 (97 linears). 11.05 -> 9.86 ms (101.4 tok/s). ACCEPTED.
  cuBLAS 5.47 -> 3.16 ms/token, kernels 2872 -> 2108/token.
- E3 inductor combo_kernels=True: 9.86 -> 9.64 ms (kernels 2108 -> 1905). KEPT (small).
- E4 compile mode max-autotune (M=1 mm decomposed into fused Triton reductions, ~5 TB/s on fused qkv): 9.64 -> 8.87 ms (112.7 tok/s), kernels 1548/token. ACCEPTED.
- E5 E4 without the GEMV kernel (fusion only): 9.13 ms. REJECTED (GEMV kept for N<1024).
- E6 GPU-driven expert offload (gpu:<fraction>): banks in pinned host memory read by the GPU over PCIe (UVA), device pool
  of 25% of 24576 expert slots, global LRU planned on the GPU (offload_plan kernel), misses read by the matvec itself and
  installed into their slot: no host round trip, whole decode step in one CUDA graph. Kernel-tested bit-exact vs resident.
  ours:0.25+compile (PR host-driven store) 3.8 tok/s -> gpu:0.25+compile 67.0 tok/s (bench, 4 prompts x 128; with E2-E4).
  ACCEPTED. Peak memory 137 GiB (to investigate: should be ~full - 48 GiB).
- E7 lookahead prefetch (router(L+1) on layer L's input, side stream plan+install inside the CUDA graph): 67.0 -> 59.4 tok/s.
  REJECTED: wrong guesses evict useful experts and compete for PCIe; per-prompt rate decays (66.7 -> 55.2).
- Memory: the 137 GiB peak was accelerate's detach_hook materializing every offloaded bank on the GPU (64.6 GiB) before
  the store replaced it; steady state is 74 GiB (57 non-expert + 17 pool). Fixed: hooks dropped without detach.
- Measurement hygiene (2026-11-23 ~02:00): the machine is shared (other users' jobs on GPUs 0,2,3,4,6,7). Runs on GPU0
  after ~01:56 may be contaminated (incl. E7). All measurements now go through q.sh on GPU1 (refuses if any other
  process holds the GPU, flock-serialized, records state at end). CPU-heavy compiles in parallel also perturb replay.
- E8 fused router kernel (logits + softmax + top-k + renorm, one launch, last-block-done): 23.8 -> 8.8 us/call,
  expert sets identical to torch 500/500 (order among bf16 ties differs). ACCEPTED (kept in decode linears).
- E9 inductor PDL (TORCHINDUCTOR_ENABLE_PDL=1): full residency 8.83 -> 8.13 ms (with E8). ACCEPTED (single run on
  then-contended GPU2; re-verify).
- Small-read limit (measured): one graph-launched kernel reading 16 MB takes >= ~12 us (1.3 TB/s); 2 MB ~8 us. The expert
  kernels (12-15 us, 9-18 MB) are at that limit: v2 (preloaded route metadata) and exact float trick (no I2F) gave 0%.
- gpu:0.25+compile, clean GPU1, PDL + router + lookahead: 69.8 tok/s (14.3 ms). Full residency same levers ~8.1 ms
  -> streamed misses cost ~6 ms/token: per-layer inline PCIe reads are latency-bound (1 missed expert: 17.5 GB/s).

## STATUS / ceiling check vs the 300 tok/s target (3.33 ms/token)
- PCIe floor at 0.25: 64.6 misses/token (best practical policy, 4 prompts x 255 tokens) x 2.76 MB / 53 GB/s = 3.4 ms,
  i.e. ~300 tok/s only with PCIe 100% busy AND fully hidden behind compute.
- Compute floor: 9.82 GB/token / 5.93 TB/s = 1.66 ms; with the measured small-kernel limit (~96 expert kernels x
  >= 8-12 us, dense at ~80%) the practical full-residency floor is ~3.5-4 ms on this kernel structure.
- So 300 tok/s at 0.25 needs both a megakernel-class compute path (< 3.3 ms) and perfect PCIe overlap; neither exists yet.
- E7b lookahead prefetch, clean A/B on GPU1 (PDL + router): nola 68.97 | lookahead 71.86 | nola 71.56 tok/s.
  INCONCLUSIVE (+0.4% over the nearer control, inside noise). Kept off (simpler, no LRU pollution).
- E10 split down-bank install (side stream during gate/up): 71.3 | 98.5 | 71.6 tok/s BUT greedy output is garbage
  (repeating 3-token loop; 0/128 tokens match the controls). INVALID / REJECTED until fixed. Harness gap: a single-config
  run compares against itself; A/B runs must compare tokens against the control (now checked from runs.jsonl).
- E10 root cause: experts_install_down returned None with no declared mutation -> inductor dropped it as dead code; the
  down projection read stale slots (fast + garbage). Fix: the install returns a token the wait consumes (data edge).
  Fixed and correct (compiled split vs eager: 109/128 identical, normal compile drift): 69.9 tok/s vs ~71.5 controls.
  REJECTED: no gain; the 98.5 was the cost of the skipped copies, i.e. the down-bank miss traffic is ~3.9 ms/token.
- Double-check of lookahead (trace, decode window): prefetch install kernels are captured and 75% overlapped -> the
  mechanism works. The bench is PCIe-bytes-bound: ~92 misses/token (eager count; 4 x 128-token prompts, cold pool) =
  ~255 MB/token ~ 5 ms; prefetch trades stall misses for extra traffic (k=10: 45 stall + 81 prefetched; k=4: 70.6 + 26.2).
  Offload fixed cost with zero misses: gpu-nola:1.0 warm 9.91 ms vs full 8.67 ms (+1.24 ms).
- E11 seed the pool from the prefill (each layer's most-routed prompt experts copied device->device from the staged
  bank): eager misses/token 91.7 -> 70.7 (-23%), no extra PCIe. A/B queued.
- E11 first A/B: candidate died on a Hub network error (no result); controls 71.55 / 71.56. Rerun offline (HF_HUB_OFFLINE=1).
- E11 seed A/B (offline): nola 72.10 | seed 71.16 | nola 70.60 tok/s; correct (128/128 vs eager). REJECTED (inside noise).
  Re-derived split from bench numbers: full+compile 9.07 ms, gpu:1.0 warm prompts ~11.9 ms, gpu:0.25 ~14.1 ms ->
  misses ~2 ms, offload path fixed cost ~2.8 ms (bench), compute 9 ms. Misses are not the main term any more.

## Bench-setting profile (distinct prompts, gpu-nola:0.25+compile vs full+compile)
- offload expert kernels with real misses: gate_up 2.84 + down 1.90 ms/token vs resident 0.76 + 0.73 -> inline PCIe miss
  reads cost ~3.25 ms/token GPU time (+0.36 ms plan). Full-residency decode kernels ~7.4 ms/token.

## basalt (jesdga95/basalt, MIT; read-only survey) and MTP
- basalt runs this exact model (src/models/qwen4exp). Its Flash-Next numbers are all with MTP drafting (RTX PRO 6000
  resident 489-539, 5090 362-395, 5090 experts spilled to RAM 205-242 tok/s, the RAM experts computed on the CPU).
- transformers refuses assisted generation for stateful models (GDN): MTP needs (1) the qwen4exp MTP head (pre-mixer 4-stream
  residual -> norm -> fc_hidden per stream, + fc_embedding(norm(embed)) broadcast, full QSA layer with its own 512-expert MoE
  and KV, own mixer, shared lm_head; ref basalt decode/mtp/forward.cpp), (2) GDN conv/recurrent state rollback for
  verify, (3) PLE n-gram history + KV/indexer crop, (4) multi-token offload plan (union of routes), (5) compiled verify graph.
- At 0.25, MTP alone does not lift the PCIe ceiling: a verify over ~3 tokens misses the union of their experts.

## Host compute tier (measured)
- host DRAM: 202 GB/s read+write (112 threads, both sockets), bf16 GEMV 338 GB/s -> 70 missed experts (193 MB) in
  ~0.6-1 ms vs ~3.6 ms of inline PCIe reads. Per-layer GPU<->CPU round trip (mapped memory + doorbell) est. 25-30 us
  -> ~1.3 ms/token serialized: projected 14.1 -> ~11.8 ms/token (~85 tok/s) from this lever alone.

## Roadmap to the 300 tok/s target (each multi-day)
1. Host compute tier for misses (GPU route kernel -> mapped doorbell -> AVX-512 4-bit affine experts on pinned worker
   threads -> mapped result + flag; GPU computes resident experts concurrently).
2. Persistent MoE kernel per layer + PDL (basalt pk_moe.cuh, pdl.hpp), then persistent GDN layer (mid_chain/hc_fused).
3. MTP drafting with GDN state rollback and a compiled multi-token verify (basalt decode/mtp, gdn_window).
- E12 host compute tier: CPU workers (AVX-512, 4-bit affine g64, pinned to node-0 cores) compute the missed experts from
  the pinned banks; GPU posts misses + x into a mapped mailbox before the resident path, a wait kernel adds the result.
  Kernel test: rel err 2.4e-3 vs GPU (bf16 rounding); round trip with 32 threads +40 us (2 misses), +82 us (4), vs ~53 us
  to DMA ONE expert over PCIe. >32 threads collapses (shared machine, spinning workers preempted). Eager model: 32/32
  tokens identical to the GPU path. Compiled A/B queued.
- sweep 1: full 16.3, full+compile 111.3, ours:0.25 eager 2.6 tok/s; ours:0.25+compile (PR store + decode-linears/router kernels) -> illegal memory access (to debug); rest re-queued one process per config.
- E12 host tier A/B (compiled, GPU1): nola 70.00 | host 74.59 | nola 70.73 tok/s; per prompt host [76.1, 74.3, 72.7, 73.7]
  vs nola [75.6, 69.9, 67.9, 66.8]; tokens vs eager 79/128 (= control). ACCEPTED (+5.5%). Best gpu:0.25+compile = 74.6.
  Smaller than projected (~0.85 vs ~2.3 ms/token): the pool is static after seeding (more misses than LRU) + per-layer
  round trip. Next: refresh the pool in the background (CPU-computed misses also staged into the pool off the critical path).
- Sweep (one process per config): gpu-nola:0.25+compile 71.9 | gpu:0.25+compile 71.7 | seed 72.8 | host 65.3 (vs 74.6 in
  its A/B) | gpu-nola:1.0+compile 89.5 | full eager 16.1 | full+compile 111.3 | ours:0.25 eager 2.6 | ours:0.25+compile CRASH
  (illegal memory access, reproducible). Peak GiB in the sweep is the full reference's (each process ran full first).
- E13 host + background pool refresh (side stream, victims limited to slots no later layer reads): correct (eager 64/64
  vs full). A/B host 56.7 | refresh 69.5 | host 74.4: INCONCLUSIVE, the identical controls differ by 31% (host-tier
  workers share cores with other users' load spikes; 5-min load avg reached 57).
- E14 persistent per-layer MoE kernel (cooperative, grid barriers, L2 prefetch of down banks during gate/up): correct
  (1.9e-4 vs split); no-miss 48 us vs split 45 us (first version 127 us: serial rows + act recomputed per down row).
  With host misses it is ~150 us slower than split (interaction to debug). Not integrated.
- Host-tier diagnosis: the CPU round trip is the critical path (2 misses: 85 us vs GPU 43 us). Fixes: act computed once
  across threads (was every thread, scalar exp: ~12 us each), experts-outer down loop, 40 node-local threads:
  2 misses 84.7 -> 70.7 us, 10 misses 300 -> 176 us. Pinned banks had landed on node 1 (main thread on cpu 190) while
  GPU1 / workers are on node 0: q.sh now binds the process to the GPU's node with NUMA=1.
- E15 NUMA-bound (node 0) A/B, host tier with the CPU fixes (40 threads): nola 70.17 | host 69.72 | nola 71.83 tok/s,
  tokens vs eager 100 / 128 / 100. REJECTED this run (inside noise; earlier un-bound A/B gave host 74.6 vs 70.0/70.7).
  The host tier's gain does not reproduce reliably on this shared machine.

## STATUS (2026-11-23 ~10:15)
Best verified gpu:0.25+compile 74.6 tok/s (one A/B; a later A/B did not reproduce the gain: ~70-72 is the robust number).
full+compile 111 tok/s. Target 300 not reached; see the ceiling notes above and the roadmap (persistent kernels, MTP).
- E16 MTP head (qwen4exp, ref basalt decode/mtp/forward.cpp) built from the model's own classes in mtp/mtp_accept.py:
  pre-mixer 4-stream residual -> grouped rms (10240 weight, group 2560) -> fc_hidden per stream, + fc_embedding(rms(embed(x[t+1])))
  broadcast -> 1 indexed-attention decoder layer (hyper-connections, 512-expert 4-bit MoE) -> own mixer -> shared lm_head.
  Loads with no missing / unexpected keys. Teacher-forced depth-1 acceptance on the bench's 4 prompts x 128 greedy tokens:
  0.837 (425/508), position convention t or t+1 irrelevant. Projected ~1.45-1.5x at 0.25 with k=1 (1.84 tokens / ~1.25x step).
  Remaining work: GDN state rollback (per-token recurrent/conv snapshots in the verify step), KV/indexer/position/PLE crop,
  multi-token offload plan (dedupe routes), compiled T=2 verify graph with on-device acceptance.
- E17 eager MTP speculative decode (mtp/mtp_spec.py): K=2 chained drafts, verify [x, d1, d2] in one main forward, GDN
  per-token recurrent states captured (recurrent rule stepped per token during verify), rollback = crop every layer by
  the rejected count (+ conv/PLE histories via the cache's past recording; fixed crop on unused state slots) and restore
  the recurrent state at the accepted index; MTP catch-up rows over the accepted tokens.
  Correctness: forced-rejection 96/96 and oracle drafts 96/96 identical to greedy on prompt 1 (rollback + verify exact);
  prompt 2 diverges at a bf16 near-tie (T=3 batched forward vs T=1) even with forced rejection: numerics, not a bug.
  Speed (eager, full residency): 2.18-2.53 tokens/verify step, 23.8-27.6 tok/s vs plain eager 14.5 (~1.7-1.9x).
  Next: compiled static-cache verify + MTP graphs, on-device rollback, offload T=K+1 path.
- E18 compiled MTP engine (mtp/mtp_engine.py, static caches, CUDA graphs, K=2): fixed per-step recompiles (one
  max_cache_len for every call, a distinct function object per MTP row count, no Python state in sibling slices under
  compile: AOT CSE merges the fused matmul), GEMV for 1-4 rows, max-autotune + PDL. Full residency: 126.0 / 146.8 / 139.9
  / 133.2 tok/s on the 4 bench prompts (2.37-2.78 tokens/step) vs full+compile 111 -> ~+22%. Per step: drafts 1.1 ms,
  verify 16.3 ms (1.9x a decode step), rollback + catch-up 1.5 ms. Next: offload for T=2-4 (duplicate-aware plan).
- E19 compiled MTP speculative at 0.25 (gpu-nola LRU, duplicate-aware T=1..4 offload plan, kernel test exact):
  82.5 / 77.7 / 76.8 tok/s (prompts 2-4; prompt 1 58.5 with a 13.5 ms rollback outlier while warming), 2.5-2.8
  tokens/step. Verify 30 ms at 0.25 vs 16.3 ms resident: ~14 ms/step of misses (union of 3 tokens' experts) = ~5.5 ms per
  emitted token. Best 0.25 so far: ~77-82 tok/s (engine harness). MTP head experts (1.35 GB, ~2% of expert bytes) resident,
  outside the 0.25 budget. Next: host tier for multi-token calls (up to 4 rows, (token, expert, weight) triples).
- E20 multi-token host tier (mailbox up to 4 rows + token index per route; kernel test rel err 2.5e-3 vs resident) inside
  the compiled MTP engine at 0.25: 68.9 / 86.2 / 76.9 / 82.7 tok/s vs GPU-only speculative 58.5 / 82.5 / 77.7 / 76.8: flat
  (verify ~27-30 ms either way). Correct (eager-spec agreement 14-128/128, bf16 near-ties).

## STATUS (2026-11-23 ~13:30)
| config (bench prompts, 128 tokens) | tok/s |
|---|---:|
| full+compile (bench) | 111 |
| MTP K=2 speculative, full residency (engine) | 126-147 |
| gpu-nola:0.25+compile (bench) | ~70-72 |
| MTP K=2 speculative at 0.25 (engine) | ~77-86 |
| ours:0.25+compile, PR store + kernels (bench) | 30.7 (was 3.8, crash fixed) |
300 tok/s at 0.25 not reached. Verify at 0.25 costs ~14 ms/step more than resident (union of 3 tokens' experts over
PCIe or CPU); compute path 16 ms/verify, 8.7 ms/decode. Remaining levers (multi-day): persistent per-layer kernels,
parallel GDN verify (no per-token loop), multi-token fused router, larger expert budget.
- E21 draft depth at 0.25: K=1 78.6 / 71.9 / 70.8 / 70.1 tok/s (1.7-1.9 tokens/step, verify 20-25 ms) < K=2 (77-86).
  K=3 not measured: rollback assumes every conv-like state is at least T=4 wide (the PLE short-conv / n-gram history
  are narrower); would need the verify's new inputs captured. K=2 kept.
- Plateau per the skill (3 consecutive levers inside noise or worse at 0.25: E20 host tier in spec, E21 K=1, E15).
- E22 multi-token fused router (T<=4; expert sets identical to torch 200/200): full residency verify 16.3 -> 16.1-16.4 ms,
  125.0 / 146.5 / 137.0 / 126.7 tok/s (inside noise). At 0.25: 73.9 / 81.6 / 63.7 / 67.2 tok/s (noisier run, verify 25-32 ms).
  REJECTED as a win (kept: correct and not slower at full residency). 4th consecutive lever inside noise.
- Budget curve (spec K=2): 0.5 run contaminated (another user's 60 GB job started on GPU1 mid-run: 51.9 / 128.2 / 56.2 /
  113.4 tok/s), 1.0 run OOM against it. As of 13:42 every GPU of the node holds a ~60 GB job of another user
  (michael, a multi-GPU bench); no clean measurement possible until it ends.
- Budget curve, MTP K=2 compiled (clean GPU1, no other process at end), tok/s on the 4 bench prompts:
  | pool | tok/s | mean |
  |---|---|---:|
  | gpu-nola:0.25 | 82.5 / 77.7 / 76.8 (prompt 1 warm-up 58.5) | ~79 |
  | gpu-nola:0.5 | 108.5 / 132.7 / 118.7 / 122.2 | ~120 |
  | gpu-nola:1.0 (offload machinery, every expert pooled) | 109.9 / 126.3 / 109.9 / 122.5 | ~117 |
  | full residency (no offload machinery) | 125.0 / 146.5 / 137.0 / 126.7 | ~134 |
  0.5 already reaches the 1.0 rate: above ~0.5 the step is compute-bound; the offload machinery itself costs ~13%.
- E23 expert-kernel rewrite (kbench/, kernel op `qmv4_tc6`). Microbench at real shapes (random ids per call, CUDA graph):
  old scalar kernels reach 1.2-1.45 TB/s (gate_up) / 0.6-1.0 TB/s (down); the floor for a random-expert gather is ~4 TB/s
  at 10 experts and ~6 TB/s at 30 (kbench/sa/ld_sa.cu; ldg / cp.async / TMA bulk all within 10%). Ablations
  (kbench/sa/*): per-lane scattered ldg of weights + scales was the cost, not the math; MMA vs FMA is noise.
  v6 = per block one 16-row tile of every route staged with cp.async (codes + scales + activations, one round trip),
  activations permuted to bf16 pair order in smem, codes dequantized as bf16 (128 + c) by lop3, mma.m16n8k16.
  Same op handles streamed misses (reads the host bank over UVA, writes the staged tile into its pool slot).
  Resident gate_up+down per layer: T=1 30.4 -> 22.4 us, T=3 69.3 -> 44.9 us, T=4 85.1 -> 53.0 us (rel err <= 1.3e-4
  vs the scalar kernels; CPU reference 1.5e-3 = bf16 output rounding). Streamed: pools bit-identical to qmv4_offload's.
  End-to-end A/B (MOE_TC6=0 vs 1, same build, 6 guarded GPUs in parallel, clean at end):
  | config | scalar | tc6 |
  |---|---:|---:|
  | full+compile (bench) | 110.3 (9.07 ms) | 117.2 (8.53 ms) +6.3%, 128/128 tokens identical |
  | MTP K=2 full residency | 121.5/147.1/123.7/109.1 | 133.1/152.3/126.9/123.4 (~+6%) |
  | gpu-nola:0.25+compile | 65.4 | 65.9 (noise) |
  | MTP K=2 at 0.25 | 84.7/80.3/74.4/74.7 | 79.0/80.2/76.0/72.6 (noise) |
  0.25 absolute is ~5 tok/s below the earlier runs on both arms (4 offload jobs sharing host memory / PCIe at once).
  Kept (default on). At 0.25 the expert kernels are not the limit: the miss traffic is (THEORY.md bound).

## Goal (2026-11-23 evening): misses cost nothing (0.25 at the full-residency rate), latest torch / kernels
- E24 host tier v2 (kernel ops host_tier2_* / host_post2 / host_wait2 / host_fetch2 / host_hint2 / host_guess2 / host_refresh2,
  csrc/host_tier2.cpp). Isolated handoff harness kbench/host_rt.py, kbench/host_side.py (48 layers, random experts in 4.2 GB
  pinned banks, CUDA graph). Findings, in order:
  - v1 per call: 1 miss 65-75 us/layer (CPU ~43 us at ~64 GB/s + ~22 us GPU-side), 10 misses ~225 us.
  - CPU kernel: codes -> bf16 by bit tricks + vdpbf16ps. Single core, L2-hot (kbench/cpu/dot_bench.cpp): v1-style 8.9 GB/s,
    LUT vpermw same, magic 128+c with (k, k+4) nibble pairs + 4-row interleave 13.1 GB/s/core (port 0/5 bound).
    Earlier 260 ns/row reading was a cold-clock artifact; and a scale-permute bug (sf[0..15] only) caught by fp64 reference.
  - fixed costs: 3 barriers with 55-thread fetch_add ~3-5 us each -> dissemination barrier; gate+act merged (thread owns
    k-range of both halves) -> 1 barrier before down. 1 miss: 65 -> 33 us/layer (serial), CPU 12-15 us.
  - GPU side: post 5.5 us, wait 5.0 us, hint 4.1 us of PCIe round trips on the critical path -> side stream (post, poll,
    copy result into a static buffer; main joins before the add): 1 miss +3 us/layer, 2 +6, 3 +13, 5 +25, 10 +57 with
    oracle hints (each thread stages its own row slices of hinted experts into L2 during attention).
  - real per-layer misses at 0.25 LRU (miss_dist.py, 4 prompts x 128): 76.8/token, histogram 0:32% 1:28% 2:18% 3:10%
    4:6% 5:3% 6+:4%; guess top-10 covers 52% of misses, top-16 68% (misses are the surprising experts).
  - bug found: the store's tc6 flag read the modules' banks after they were replaced by empty placeholders -> tc6 was never
    used by the offload store (the E23 0.25 A/B measured the scalar path on both arms).
  - first compiled host-v2 run (static pool): 66 tok/s, 45/48 layers call the CPU (~3 misses each), guess on main stream
    0.6 ms/token; 41/128 tokens vs eager (suspect: declared mutations of side-stream state functionalized -> copy-back
    race). Fixed: state internal to the store, ordering via data edges only -> 128/128 vs full.
  - static pool, tc6 on, guess on side stream: hint16 69.1 (node 0) | hint0 72.5 (node 1) -- confounded by node; A/Bs are
    crossed across sockets from now on.
  - pool refresh by the copy engine: GPU plans LRU victims (layers <= current, unused this token), key pending (-2), ring in
    the mapped mailbox; CPU installer thread cudaMemcpyAsync host bank -> slot, done counter; GPU commits table[key] = slot
    only if owner[slot] == key. kbench/test_refresh.py: 935 installs, all committed entries byte-exact, none stale.
