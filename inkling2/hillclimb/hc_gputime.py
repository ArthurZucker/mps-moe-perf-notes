"""Low-overhead GPU timeline of decode: every committed command buffer's GPUStartTime / GPUEndTime (completion handlers
via the mps-events extension, HC_COMMIT_TIMED=1 with HC_COMMIT_EVERY=N), against the host step time. Answers: how long
is the GPU busy per token, how long does it sit idle between command buffers, and how late does it start after a commit.

    HC_COMMIT_EVERY=24 HC_COMMIT_TIMED=1 PYTHONPATH=... python hc_gputime.py configs/<name>.json [tokens]
"""
import json, os, statistics, sys, time
os.environ.setdefault("HC_COMMIT_TIMED", "1")
import torch
import hc_bench
from hc_bench import gpu_busy, on_ac
if not on_ac() or gpu_busy() > 15:
    sys.exit(f"refusing to run: battery or busy GPU ({gpu_busy()}% before the run, e.g. mediaanalysisd / a browser)")

cfg = json.load(open(sys.argv[1])); tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 96
model, tok, ids, gen = hc_bench.load(cfg)
ext = hc_bench.install_periodic_commit.keep[1]
stamps = []
class S:
    def put(self, _): stamps.append(time.perf_counter())
    def end(self): pass
with torch.inference_mode():
    for _ in range(2): model.generate(ids, max_new_tokens=16, min_new_tokens=16, **gen)
    torch.mps.synchronize(); ext.timings()
    model.generate(ids, max_new_tokens=tokens, min_new_tokens=tokens, streamer=S(), **gen)
    torch.mps.synchronize(); time.sleep(0.2)
t = sorted(ext.timings())  # (gpu_start, gpu_end, host_commit)
steps = sorted(b - a for a, b in zip(stamps[2:], stamps[3:]))
step = steps[len(steps) // 2]
t = t[len(t) // 8:]  # drop the prefill / first steps
span = t[-1][1] - t[0][0]
busy, end, gaps = 0.0, t[0][0], []
for s0, e0, *_ in t:
    if s0 > end: gaps.append(s0 - end)
    busy += max(0.0, e0 - max(s0, end)); end = max(end, e0)
lat = sorted(r[0] - r[2] for r in t)
n_tok = span / step
print(f"{cfg['name']} commit_every={os.environ.get('HC_COMMIT_EVERY')}: host step {step * 1e3:.2f} ms | {len(t) / n_tok:.1f} command buffers/token | "
      f"GPU busy {busy / n_tok * 1e3:.2f} ms/token ({100 * busy / span:.0f}%) | GPU idle {sum(gaps) / n_tok * 1e3:.2f} ms/token "
      f"(median gap {statistics.median(gaps) * 1e6 if gaps else 0:.1f} us) | commit->GPU start median {lat[len(lat) // 2] * 1e6:.0f} us", flush=True)
