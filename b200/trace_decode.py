"""Decode-only window of a chrome trace (after the last prefill expert kernel): kernels per stream, where the offload
plan / install / prefetch kernels run, and how much install time overlaps other kernels."""
import json, sys, collections
ev = [e for e in json.load(open(sys.argv[1]))["traceEvents"] if e.get("cat") == "kernel"]
t0 = max(e["ts"] for e in ev if "_gather_qmv_kernel" in e["name"] or "Memcpy" in e["name"] and e["dur"] > 1000)
dec = [e for e in ev if e["ts"] > t0]
print(f"decode window: {len(dec)} kernels, {(max(e['ts']+e['dur'] for e in dec) - min(e['ts'] for e in dec))/1e3:.1f} ms")
by = collections.defaultdict(list)
for e in dec: by[e["args"].get("stream")].append(e)
for s, es in sorted(by.items(), key=lambda x: -len(x[1]))[:6]:
    c = collections.Counter(e["name"].split("(")[0][:36] for e in es)
    print(f" stream {s}: {len(es)} kernels; plan {c['offload_plan_kernel']}, install {c['offload_install_kernel']}; top {c.most_common(3)}")
inst = sorted((e["ts"], e["ts"] + e["dur"], e["args"].get("stream")) for e in dec if "install" in e["name"])
oth = sorted((e["ts"], e["ts"] + e["dur"], e["args"].get("stream")) for e in dec if "install" not in e["name"])
ov = 0
for a, b, s in inst:
    for c, d, s2 in oth:
        if d <= a: continue
        if c >= b: break
        ov += min(b, d) - max(a, c)
print(f"install kernels {len(inst)}, time {sum(b - a for a, b, _ in inst)/1e3:.2f} ms, overlapped by other kernels {ov/1e3:.2f} ms")
