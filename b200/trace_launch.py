import json, sys, bisect
t = json.load(open(sys.argv[1]))
gl = sorted([e for e in t["traceEvents"] if e.get("name") == "cudaGraphLaunch"], key=lambda e: e["ts"])
k = sorted([e for e in t["traceEvents"] if e.get("cat") == "kernel"], key=lambda e: e["ts"]); ts = [e["ts"] for e in k]
d = sorted(e["dur"] for e in gl)
spans = []
for a, b in zip(gl, gl[1:]):
    i0 = bisect.bisect_left(ts, a["ts"]); i1 = bisect.bisect_left(ts, b["ts"])
    if i1 > i0: spans.append((b["ts"] - a["ts"], max(e["ts"] + e["dur"] for e in k[i0:i1]) - k[i0]["ts"], k[i0]["ts"] - a["ts"]))
spans.sort()
m = spans[len(spans) // 2]
print(f"{sys.argv[1]}: {len(gl)} launches, cudaGraphLaunch CPU median {d[len(d)//2]:.0f} us; median launch-to-launch {m[0]:.0f} us, GPU span {m[1]:.0f} us, launch->first kernel {m[2]:.0f} us")
