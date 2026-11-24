import json, sys, collections, heapq
d = json.load(open(sys.argv[1])); L, E = d["layers"], d["num_experts"]
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
cap = int(frac * L * E)
seq = []  # per decode token: list of (layer, expert) in forward order
for r in d["runs"]:
    for step in r["decode"]:
        seq.append([(l, e) for l, es in enumerate(step) for e in sorted(set(es))])
T = len(seq); flat = [k for s in seq for k in s]
def lru():
    c = collections.OrderedDict(); m = 0
    for k in flat:
        if k in c: c.move_to_end(k)
        else:
            m += 1; c[k] = 1
            if len(c) > cap: c.popitem(last=False)
    return m
def lfu(decay=None):
    cnt = collections.Counter(); c = set(); m = 0; heap = []
    for i, k in enumerate(flat):
        cnt[k] += 1
        if k in c: continue
        m += 1
        if len(c) < cap: c.add(k); continue
        # evict min count (lazy scan: fine for sim size)
        v = min(c, key=lambda x: cnt[x]); c.remove(v); c.add(k)
    return m
def belady():
    nxt = [0] * len(flat); last = {}
    for i in range(len(flat) - 1, -1, -1):
        nxt[i] = last.get(flat[i], 1 << 60); last[flat[i]] = i
    c = {}; h = []; m = 0
    for i, k in enumerate(flat):
        if k in c:
            c[k] = nxt[i]; heapq.heappush(h, (-nxt[i], k)); continue
        m += 1
        if len(c) >= cap:
            while True:
                negn, v = heapq.heappop(h)
                if v in c and c[v] == -negn: del c[v]; break
        c[k] = nxt[i]; heapq.heappush(h, (-nxt[i], k))
    return m
def static_top():  # oracle static set: most frequent experts over the whole trace, pinned
    cnt = collections.Counter(flat); keep = set(k for k, _ in cnt.most_common(cap))
    return sum(1 for k in flat if k not in keep)
routes = len(flat)
print(f"tokens {T}  unique-routes/token {routes/T:.1f}  cap {cap} slots ({frac:.0%})")
for name, f in [("LRU", lru), ("LFU", lfu), ("static oracle top-freq", static_top), ("Belady OPT", belady)]:
    m = f(); print(f"  {name:24s} misses/token {m/T:6.1f}  hit {1-m/routes:.3f}  PCIe MB/token {m/T*2.7648:6.1f} -> PCIe-bound ceiling {53e3/(m/T*2.7648):6.0f} tok/s")
