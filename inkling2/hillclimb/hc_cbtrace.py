"""Passive GPU timeline of an unmodified decode run: every command buffer the process commits (torch's stream, MPSGraph,
copies), with GPU start / end. Reports GPU busy / idle per token, command buffers per token, the gap distribution, and
whether gaps sit at the token boundary (the longest command buffer per token marks the step's body)."""
import json, os, statistics, sys, time
from hc_bench import gpu_busy, on_ac
if not on_ac() or gpu_busy() > 15:
    sys.exit(f"refusing to run: battery or busy GPU ({gpu_busy()}%)")
import torch
import hc_bench
from torch.utils.cpp_extension import load as load_ext

cfg = json.load(open(sys.argv[1])); tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 96
model, tok, ids, gen = hc_bench.load(cfg)
here = os.path.expanduser("~/Work/hub-kernels/mps-events")
ext = load_ext("_mps_events", [os.path.join(here, "mps_events.mm")], build_directory=os.path.join(here, ".build", f"py{sys.version_info[0]}{sys.version_info[1]}"),
               extra_cflags=["-std=c++20", "-O2", "-D_DARWIN_C_SOURCE", "-include", "sys/types.h"],
               extra_ldflags=["-framework", "Metal", "-framework", "Foundation", "-framework", "QuartzCore"], verbose=False)
print("swizzled class:", ext.trace_all(False), flush=True)
with torch.inference_mode():
    for _ in range(2): model.generate(ids, max_new_tokens=tokens, min_new_tokens=tokens, **gen)  # same length: no recompile
    torch.mps.synchronize(); ext.timings(); ext.trace_all(True)
    t0 = time.perf_counter(); model.generate(ids, max_new_tokens=tokens, min_new_tokens=tokens, **gen); torch.mps.synchronize()
    wall = time.perf_counter() - t0; ext.trace_all(False); time.sleep(0.2)
t = sorted((r for r in ext.timings() if r[1] > r[0] > 0), key=lambda r: r[0])
t = t[len(t) // 6:]  # past the prefill
span = t[-1][1] - t[0][0]
busy, end, gaps = 0.0, t[0][0], []
for s0, e0, *_ in t:
    if s0 > end: gaps.append(s0 - end)
    busy += max(0.0, e0 - max(s0, end)); end = max(end, e0)
n_tok = tokens * (5 / 6)  # approximate tokens in the window
durs = sorted(e0 - s0 for s0, e0, *_ in t)
print(f"{cfg['name']}: wall {wall / tokens * 1e3:.2f} ms/token incl. prefill | window {span * 1e3 / n_tok:.2f} ms/token | "
      f"GPU busy {100 * busy / span:.1f}% | idle {sum(gaps) / n_tok * 1e3:.2f} ms/token in {len(gaps) / n_tok:.1f} gaps/token "
      f"(median {statistics.median(gaps) * 1e6 if gaps else 0:.0f} us, max {max(gaps) * 1e6 if gaps else 0:.0f} us) | "
      f"{len(t) / n_tok:.1f} command buffers/token, durations median {durs[len(durs) // 2] * 1e6:.0f} us, max {durs[-1] * 1e3:.2f} ms", flush=True)
