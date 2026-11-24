import json, sys, collections
t = json.load(open(sys.argv[1])); ev = t["traceEvents"]
k = sorted([e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"], key=lambda e: e["ts"])
gl = sorted([e for e in ev if e.get("name") == "cudaGraphLaunch"], key=lambda e: e["ts"])
per_all, busy, nl = [], collections.defaultdict(float), 0
for g0, g1 in zip(gl[6:12], gl[7:13]):
    kk = [e for e in k if g0["ts"] <= e["ts"] < g1["ts"]]
    marks = [e["ts"] for e in kk if "tc6::kernel<2560" in e["name"]]
    per_all += [y - x for x, y in zip(marks, marks[1:])]
    for e in kk:
        if marks[1] <= e["ts"] < marks[-1]: busy[e["name"][:60]] += e["dur"]
    nl += len(marks) - 2
per_all.sort()
print(f"{sys.argv[1]}: layer period p50 {per_all[len(per_all)//2]:.1f} p90 {per_all[int(.9*len(per_all))]:.1f} mean {sum(per_all)/len(per_all):.1f} us; kernel time/layer {sum(busy.values())/nl:.1f}")
for n, v in sorted(busy.items(), key=lambda kv: -kv[1])[:6]: print(f"  {v/nl:6.1f} us  {n}")
