"""Overlap of install kernels with other kernels, replay streams only (excluding the eager/default stream)."""
import json, sys, collections
ev = [e for e in json.load(open(sys.argv[1]))["traceEvents"] if e.get("cat") == "kernel"]
by = collections.defaultdict(list)
for e in ev: by[e["args"].get("stream")].append(e)
eager = max(by, key=lambda s: len(by[s]))
rep = [e for s, es in by.items() if s != eager for e in es]
inst = sorted((e["ts"], e["ts"] + e["dur"]) for e in rep if "install" in e["name"])
other = sorted((e["ts"], e["ts"] + e["dur"]) for e in rep if "install" not in e["name"])
print("replay kernels:", len(rep), " install kernels in replay:", len(inst), " total install time %.2f ms" % (sum(b - a for a, b in inst) / 1e3))
ov = 0
j = 0
for a, b in inst:
    for c, d in other:
        if d <= a: continue
        if c >= b: break
        ov += min(b, d) - max(a, c)
print("install time overlapped by other replay kernels: %.2f ms" % (ov / 1e3))
names = collections.Counter(e["name"].split("(")[0][:50] for e in rep)
print(names.most_common(8))
