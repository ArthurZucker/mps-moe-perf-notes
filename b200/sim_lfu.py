"""Realizable frequency policies at 0.25 (global pool), routing_256: per access decayed score s = s * decay^(dt tokens) + 1.
Evict the resident key with the lowest current score. Admission: a missed key is admitted only if its score beats the
victim's (otherwise the miss is served without installing)."""
import json, sys, math, collections, heapq
d = json.load(open(sys.argv[1])); L, E = d["layers"], d["num_experts"]
cap = int(float(sys.argv[2]) * L * E)
runs = [[[(l, e) for l, es in enumerate(step) for e in sorted(set(es))] for step in r["decode"]] for r in d["runs"]]
def sim(half_life, admit, per_prompt_reset=False, seed_lru_size=None):
    lam = math.log(2) / half_life
    misses = toks = 0
    score = {}; t_of = {}; res = set()
    def cur(k, t): return score.get(k, 0.0) * math.exp(-lam * (t - t_of.get(k, t)))
    for r in runs:
        if per_prompt_reset: score.clear(); t_of.clear()
        for t, step in enumerate(r):
            T = toks + t
            for k in step:
                sc = cur(k, T) + 1.0; score[k] = sc; t_of[k] = T
                if k in res: continue
                misses += 1
                if len(res) < cap: res.add(k); continue
                # victim: lowest decayed score (exact scan is slow -> sample 64 residents, like Redis' approximated LFU)
                cand = [x for _, x in zip(range(64), iter_sample())]
                v = min(cand, key=lambda x: cur(x, T))
                if admit and cur(v, T) >= sc: continue
                res.discard(v); res.add(k)
        toks += len(r)
    return misses / toks
import random
random.seed(0)
_res_list = None
def iter_sample():
    # sample from the module-level resident set (set by sim via closure trick)
    return random.sample(list(SIM_RES[0]), 64) if len(SIM_RES[0]) > 64 else list(SIM_RES[0])
SIM_RES = [None]
_orig = sim
def sim(half_life, admit):
    global SIM_RES
    lam = math.log(2) / half_life
    misses = toks = 0; score = {}; t_of = {}; res = set(); SIM_RES[0] = res
    reslist = []  # random sampling support
    pos = {}
    def add(k): pos[k] = len(reslist); reslist.append(k); res.add(k)
    def rem(k):
        i = pos.pop(k); last = reslist.pop()
        if i < len(reslist): reslist[i] = last; pos[last] = i
        res.discard(k)
    cur = lambda k, T: score.get(k, 0.0) * math.exp(-lam * (T - t_of.get(k, T)))
    for r in runs:
        for t, step in enumerate(r):
            T = toks + t
            for k in step:
                sc = cur(k, T) + 1.0; score[k] = sc; t_of[k] = T
                if k in res: continue
                misses += 1
                if len(res) < cap: add(k); continue
                cand = [reslist[random.randrange(len(reslist))] for _ in range(48)]
                v = min(cand, key=lambda x: cur(x, T))
                if admit and cur(v, T) >= sc: continue
                rem(v); add(k)
        toks += len(r)
    return misses / toks
for hl in (4, 16, 64, 256):
    for admit in (False, True):
        print(f"decayed LFU half-life {hl:4d} tokens admit={admit!s:5s}: misses/token {sim(hl, admit):6.1f}", flush=True)
