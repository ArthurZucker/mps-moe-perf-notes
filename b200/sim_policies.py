import json, sys, collections
d = json.load(open(sys.argv[1])); L, E = d["layers"], d["num_experts"]
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
cap = int(frac * L * E)
seq = [[(l, e) for l, es in enumerate(step) for e in sorted(set(es))] for r in d["runs"] for step in r["decode"]]
T = len(seq); flat = [k for s in seq for k in s]
def run(policy):
    m = 0
    for i, k in enumerate(flat):
        if not policy.access(k, i): m += 1
    return m / T
class LRU:
    def __init__(s): s.c = collections.OrderedDict()
    def access(s, k, i):
        if k in s.c: s.c.move_to_end(k); return True
        s.c[k] = 1
        if len(s.c) > cap: s.c.popitem(last=False)
        return False
class LRUK:  # LRU-2: evict the item whose 2nd-most-recent access is oldest (items seen once: oldest first)
    def __init__(s): s.h = {}; s.res = set()
    def access(s, k, i):
        h = s.h.setdefault(k, [-1, -1]); h[0], h[1] = i, h[0]
        if k in s.res: return True
        if len(s.res) >= cap:
            v = min(s.res, key=lambda x: (s.h[x][1], s.h[x][0])); s.res.remove(v)
        s.res.add(k); return False
class SLRU:  # segmented: probation (20%) + protected (80%)
    def __init__(s, pfrac=0.2): s.p = collections.OrderedDict(); s.q = collections.OrderedDict(); s.pc = int(cap * pfrac); s.qc = cap - s.pc
    def access(s, k, i):
        if k in s.q: s.q.move_to_end(k); return True
        if k in s.p:
            del s.p[k]; s.q[k] = 1
            if len(s.q) > s.qc:
                v, _ = s.q.popitem(last=False); s.p[v] = 1
                if len(s.p) > s.pc: s.p.popitem(last=False)
            return True
        s.p[k] = 1
        if len(s.p) > s.pc: s.p.popitem(last=False)
        return False
class ARC:
    def __init__(s): s.t1, s.t2, s.b1, s.b2 = (collections.OrderedDict() for _ in range(4)); s.p = 0
    def _replace(s, k):
        if s.t1 and (len(s.t1) > s.p or (k in s.b2 and len(s.t1) == s.p)):
            v, _ = s.t1.popitem(last=False); s.b1[v] = 1
        else:
            v, _ = s.t2.popitem(last=False); s.b2[v] = 1
    def access(s, k, i):
        if k in s.t1: del s.t1[k]; s.t2[k] = 1; return True
        if k in s.t2: s.t2.move_to_end(k); return True
        if k in s.b1:
            s.p = min(cap, s.p + max(len(s.b2) // max(len(s.b1), 1), 1)); s._replace(k); del s.b1[k]; s.t2[k] = 1; return False
        if k in s.b2:
            s.p = max(0, s.p - max(len(s.b1) // max(len(s.b2), 1), 1)); s._replace(k); del s.b2[k]; s.t2[k] = 1; return False
        if len(s.t1) + len(s.b1) == cap:
            if len(s.t1) < cap: s.b1.popitem(last=False); s._replace(k)
            else: s.t1.popitem(last=False)
        elif len(s.t1) + len(s.b1) < cap and len(s.t1) + len(s.t2) + len(s.b1) + len(s.b2) >= cap:
            if len(s.t1) + len(s.t2) + len(s.b1) + len(s.b2) == 2 * cap: s.b2.popitem(last=False)
            s._replace(k)
        s.t1[k] = 1; return False
for name, p in [("LRU", LRU()), ("SLRU 20/80", SLRU(0.2)), ("SLRU 50/50", SLRU(0.5)), ("ARC", ARC()), ("LRU-2", LRUK())]:
    m = run(p); print(f"{name:12s} misses/token {m:6.1f} -> PCIe {m*2.7648:6.1f} MB/token, PCIe-bound ceiling {53e3/(m*2.7648):5.0f} tok/s", flush=True)
