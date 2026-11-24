import json, sys
t = json.load(open(sys.argv[1])); ev = t["traceEvents"]
k = sorted([e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"], key=lambda e: e["ts"])
gl = sorted([e for e in ev if e.get("name") == "cudaGraphLaunch"], key=lambda e: e["ts"])
kk = [e for e in k if gl[10]["ts"] <= e["ts"] < gl[11]["ts"]]
idx = [i for i, e in enumerate(kk) if "tc6::kernel<2560" in e["name"]]
pers = sorted((kk[b]["ts"] - kk[a]["ts"], a, b) for a, b in zip(idx, idx[1:]))
per, a, b = pers[len(pers) // 2]
t0 = kk[a]["ts"]
print(f"{sys.argv[1]}: median layer {per:.1f} us")
for e in kk[a:b + 1]:
    print(f"  {e['ts'] - t0:7.1f} +{e['dur']:5.1f}  s{e['args'].get('stream')}  {e['name'][:70]}")
