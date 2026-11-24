# Ceilings: Qwen3.8-Flash-Next (ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit) decode on one B200

Measured: HBM read 5.93 TB/s (x.sum), d2d copy 3.27 TB/s, pinned H2D 55.7 GB/s (51.3 GB/s in 2.76 MB chunks),
pageable H2D 23.5 GB/s, NVLink peer->GPU0 769 GB/s.

Bytes per decode token (bf16 dense, 4-bit g64 experts):
| component | GB/token |
|---|---:|
| dense nn.Linear (36 GDN + 12 full-attn layers, hyper-connections, shared experts, router) | 7.22 |
| lm_head (248320 x 2560 bf16) | 1.27 |
| routed experts: 48 layers x top-10 x 2.765 MB | 1.33 |
| total | 9.82 |
Full-residency floor: 9.82 GB / 5.93 TB/s = 1.66 ms -> ~600 tok/s (practical at 80%: ~480 tok/s).

ours:0.25 (6144 of 24576 expert slots on device, misses read from pinned host RAM over PCIe):
| policy (sim on 4 prompts x 255 real decode tokens, global pool) | misses/token | PCIe-bound ceiling |
|---|---:|---:|
| LRU | 64.6 | 297 tok/s |
| LFU | 140.7 | 136 tok/s |
| static oracle top-freq | 128.0 | 150 tok/s |
| Belady OPT | 34.7 | 553 tok/s |
300 tok/s = 3.33 ms/token: needs <= ~64 misses/token streamed fully overlapped with ~<3 ms of compute, i.e. better than LRU
or perfect overlap. Measured bench (PR's host-driven store): 85.6 misses/token, 48 host round trips/token.

## Hard bound for ours:0.25 decode on this B200 (measured inputs, 2026-11-23)
- Expert misses at 0.25, bench prompts, cold-start LRU, equal-terms count: 73.7 / token (plain decode). A K=2 verify misses the
  union of 3 tokens' experts: ~14 ms/step extra measured, i.e. ~5.5 ms per emitted token at 2.5 tokens/step.
- PCIe host->device measured 51-55 GB/s; an expert is 2.765 MB. Plain decode: 73.7 x 2.765 MB = 204 MB/token -> 3.8 ms/token
  of transfer -> <= ~260 tok/s with ZERO compute and perfect overlap.
- Compute: the fastest measured decode with every expert resident is 8.7 ms/token (plain) / ~7.4 ms per emitted token
  (MTP K=2); memory-bandwidth floor 1.66 ms/token.
- So 300 tok/s (3.33 ms/token) at 0.25 needs BOTH transfer below the PCIe bound above (only possible by computing misses on
  the host: 204 MB at ~300 GB/s host DRAM = 0.7 ms, but the measured per-layer GPU<->CPU round trip is 70-180 us x 48)
  AND a compute path within 2x of the bandwidth floor (currently 4.5x; 2,700 kernels per verify step). Neither exists here.
