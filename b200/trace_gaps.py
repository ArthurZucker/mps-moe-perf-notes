"""From a chrome trace of compiled decode: per stream, busy time vs span; on the main stream, the idle gaps by what
runs right after them (the kernel that was waiting)."""
import json, sys, collections
t = json.load(open(sys.argv[1]))
ev = [e for e in t["traceEvents"] if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"]
by = collections.defaultdict(list)
for e in ev: by[e["args"].get("stream", e.get("tid"))].append(e)
ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 1
for s, es in sorted(by.items(), key=lambda kv: -len(kv[1])):
    es.sort(key=lambda e: e["ts"])
    busy = sum(e["dur"] for e in es); span = es[-1]["ts"] + es[-1]["dur"] - es[0]["ts"]
    print(f"stream {s}: {len(es)} kernels, busy {busy / 1e3 / ntok:.2f} ms/token, span {span / 1e3 / ntok:.2f} ms/token")
main = max(by.values(), key=len); main.sort(key=lambda e: e["ts"])
gaps = collections.defaultdict(lambda: [0.0, 0])
for a, b in zip(main, main[1:]):
    g = b["ts"] - (a["ts"] + a["dur"])
    if g > 0:
        gaps[b["name"][:60]][0] += g; gaps[b["name"][:60]][1] += 1
tot = sum(v[0] for v in gaps.values())
print(f"main-stream idle {tot / 1e3 / ntok:.2f} ms/token; largest, by the kernel that waited:")
for k, (g, c) in sorted(gaps.items(), key=lambda kv: -kv[1][0])[:12]:
    print(f"  {g / 1e3 / ntok:6.3f} ms/tok  {c / ntok:6.1f}/tok  avg {g / c:6.1f} us  {k}")
