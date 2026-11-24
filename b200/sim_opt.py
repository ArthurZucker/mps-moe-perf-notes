import json, sys, collections, heapq
d = json.load(open(sys.argv[1])); L, E = d["layers"], d["num_experts"]
frac = float(sys.argv[2]); cap = int(frac * L * E)
runs = [[[(l, e) for l, es in enumerate(step) for e in sorted(set(es))] for step in r["decode"]] for r in d["runs"]]
seq = [s for r in runs for s in r]; T = len(seq); flat = [k for s in seq for k in s]
# Belady: evict the resident item whose next use is farthest
nxt = [0] * len(flat); last = {}
for i in range(len(flat) - 1, -1, -1): nxt[i] = last.get(flat[i], 10**12); last[flat[i]] = i
res = {}; heap = []; m = 0
for i, k in enumerate(flat):
    if k in res: res[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k)); continue
    m += 1
    if len(res) >= cap:
        while True:
            nu, v = heapq.heappop(heap)
            if v in res and res[v] == -nu: del res[v]; break
    res[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k))
print(f"Belady OPT   misses/token {m / T:6.1f}")
# static oracle: best fixed set (most frequent over the whole trace) -- per-prompt oracle too
cnt = collections.Counter(flat); top = set(k for k, _ in cnt.most_common(cap))
print(f"static global-frequency set   misses/token {sum(k not in top for k in flat) / T:6.1f}")
mm = 0
for r in runs:
    f = [k for s in r for k in s]; c = collections.Counter(f); t = set(k for k, _ in c.most_common(cap)); mm += sum(k not in t for k in f)
print(f"static per-prompt oracle set  misses/token {mm / T:6.1f}")
print(f"routes/token {len(flat) / T:.1f}")
