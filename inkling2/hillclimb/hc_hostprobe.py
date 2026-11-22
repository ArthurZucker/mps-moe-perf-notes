"""Host-side timeline of the decode loop: per step, host time inside the compiled forward call (encode), inside the
stop check (blocked on the device), and everything else in generate's loop; plus which stop check is used."""
import json, os, statistics, sys, time
from hc_bench import gpu_busy, on_ac
if not on_ac() or gpu_busy() > 15:
    sys.exit(f"refusing to run: battery or busy GPU ({gpu_busy()}%)")
import torch
import hc_bench
from transformers.generation import utils as GU

cfg = json.load(open(sys.argv[1])); tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 128
model, tok, ids, gen = hc_bench.load(cfg)
log = {"fwd": [], "stop": [], "stop_sync": [], "kinds": set(), "marks": []}
orig_get = model.get_compiled_call
def get_compiled_call(cc):
    compiled = orig_get(cc)
    def timed(*a, **k):
        t = time.perf_counter(); out = compiled(*a, **k); log["fwd"].append(time.perf_counter() - t); log["marks"].append(t); return out
    return timed
model.get_compiled_call = get_compiled_call
for cls in (GU.StopCheck, GU.DeferredStopCheck):
    call = cls.__call__
    def wrapped(self, *a, _call=call, _cls=cls, **k):
        log["kinds"].add(_cls.__name__); t = time.perf_counter(); out = _call(self, *a, **k); log["stop"].append(time.perf_counter() - t); return out
    cls.__call__ = wrapped
# torch.Event is immutable: time the blocking part through the deque slots' events by wrapping DeferredStopCheck's events
orig_init = GU.DeferredStopCheck.__init__
class TimedEvent:
    def __init__(self, ev): self.ev = ev
    def record(self, *a, **k): return self.ev.record(*a, **k)
    def synchronize(self):
        t = time.perf_counter(); self.ev.synchronize(); log["stop_sync"].append(time.perf_counter() - t)
def init(self, *a, **k):
    orig_init(self, *a, **k)
    import collections
    self.slots = collections.deque((s, tk, TimedEvent(ev)) for s, tk, ev in self.slots)
GU.DeferredStopCheck.__init__ = init
if os.environ.get("HC_NO_STOP_COPY") == "1":  # probe: skip the per-step device -> host copy of the stop flag
    def call(self, unfinished_sequences, tokens, length):
        stopping = self.max_length is not None and length >= self.max_length
        return stopping
    GU.DeferredStopCheck.__call__ = call
with torch.inference_mode():
    for _ in range(2): model.generate(ids, max_new_tokens=16, min_new_tokens=16, **gen)
    for k in ("fwd", "stop", "stop_sync", "marks"): log[k].clear()
    def timed(n):
        torch.mps.synchronize(); t0 = time.perf_counter()
        model.generate(ids, max_new_tokens=n, min_new_tokens=n, **gen); torch.mps.synchronize()
        return time.perf_counter() - t0
    diffs = []
    for _ in range(3):
        short, long_ = timed(8), timed(8 + tokens)
        diffs.append((long_ - short) / tokens)
    decode_ms = statistics.median(diffs) * 1e3
    for k in ("fwd", "stop", "stop_sync", "marks"): log[k].clear()
    t0 = time.perf_counter(); model.generate(ids, max_new_tokens=tokens, min_new_tokens=tokens, **gen); torch.mps.synchronize()
    total = time.perf_counter() - t0
med = lambda xs: statistics.median(xs[4:]) * 1e3 if len(xs) > 4 else float("nan")
period = statistics.median(b - a for a, b in zip(log["marks"][4:], log["marks"][5:])) * 1e3
print(f"stop check: {sorted(log['kinds'])} | step period {period:.2f} ms | host in compiled forward {med(log['fwd']):.2f} ms | "
      f"in stop check {med(log['stop']):.2f} ms (of which Event.synchronize {med(log['stop_sync']):.2f} ms) | "
      f"rest of the loop {period - med(log['fwd']) - med(log['stop']):.2f} ms | DECODE {decode_ms:.2f} ms/token = {1e3 / decode_ms:.1f} tok/s (differential)")
