"""From a chrome trace: per-stream kernel time, whether install/prefetch kernels exist, and how much they overlap
main-stream kernels."""
import json, sys, collections
ev = [e for e in json.load(open(sys.argv[1]))["traceEvents"] if e.get("cat") == "kernel"]
by = collections.defaultdict(list)
for e in ev: by[e["args"].get("stream", e.get("tid"))].append((e["ts"], e["ts"] + e["dur"], e["name"]))
for s, ks in sorted(by.items(), key=lambda x: -len(x[1])):
    names = collections.Counter(n.split("(")[0][:40] for _, _, n in ks)
    print(f"stream {s}: {len(ks)} kernels, {sum(b - a for a, b, _ in ks)/1e3:.1f} ms; top: {names.most_common(4)}")
main = max(by, key=lambda s: len(by[s]))
mk = sorted((a, b) for a, b, _ in by[main])
inst = [(a, b) for s, ks in by.items() if s != main for a, b, n in ks]
inst_main = [(a, b) for a, b, n in by[main] if "install" in n or "offload_plan" in n]
print("install/plan kernels on main stream:", len(inst_main))
def overlap(iv, others):
    tot = 0
    for a, b in iv:
        for c, d in others:
            if d <= a: continue
            if c >= b: break
            tot += min(b, d) - max(a, c)
    return tot
side_time = sum(b - a for a, b in inst)
print(f"side-stream kernel time {side_time/1e3:.2f} ms, overlapped with main-stream kernels {overlap(sorted(inst), mk)/1e3:.2f} ms")
