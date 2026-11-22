"""Width of a compiled decode graph: parse inductor's wrapper (latest big one), build the kernel dependency DAG from the
buffers each launch reads / writes (RAW + WAR + WAW, since inductor reuses buffers), and report total kernels vs critical
path length and the level histogram (how many kernels could run side by side)."""
import collections, glob, os, re, sys
D = "/var/folders/sn/cck81hm1547_txs7j30btlsw0000gn/T/torchinductor_arthurzucker"
path = sys.argv[1] if len(sys.argv) > 1 else max((f for f in glob.glob(D + "/**/*.py", recursive=True) if os.path.getsize(f) > 300e3), key=os.path.getmtime)
s = open(path).read()
# outputs per generated kernel: params named out_ptr* / in_out_ptr* (in order) come first in the call
nouts = {}
for m in re.finditer(r"(generated_kernel_\d+) = async_compile\.metal\('\1', '''(.*?)'''", s, re.S):
    src = m.group(2); header = src[src.index("kernel void"):src.index("{")]
    nouts[m.group(1)] = len(re.findall(r"\b(?:in_out_ptr|out_ptr)\d+\b", header))
a = s.find("def partition_0") if "def partition_0" in s else s.find("def call(")
body = s[a:s.find("class Runner", a) if "class Runner" in s[a:] else len(s)]
alias = {}  # reinterpret / views: name -> base buffer
def base(n): 
    while n in alias: n = alias[n]
    return n
nodes = []  # (reads, writes, label)
for line in body.splitlines():
    t = line.strip()
    mv = re.match(r"(\w+) = (?:reinterpret_tensor|torch\.ops\.aten\.\w+\.default)?\(?(\w+)[,\.\)]", t)
    ix = re.match(r"(\w+) = (\w+)\[\d+\]$", t) or re.match(r"(\w+) = (buf\w+|arg\w+)$", t)
    if ix:
        alias[ix.group(1)] = ix.group(2); continue
    if re.match(r"(\w+) = reinterpret_tensor\((\w+)", t):
        x = re.match(r"(\w+) = reinterpret_tensor\((\w+)", t); alias[x.group(1)] = x.group(2); continue
    g = re.match(r"(generated_kernel_\d+)\(([^)]*)", t)
    if g:
        args = [x.strip() for x in g.group(2).split(",") if re.match(r"\s*(buf|arg)\w*$", x)]
        k = nouts.get(g.group(1), 1)
        nodes.append(({base(x) for x in args[k:]}, {base(x) for x in args[:k]}, "inductor")); continue
    c = re.match(r"(\w+(?:, \w+)*) = (torch\.ops\.[\w.]+|extern_kernels\.\w+)\((.*)\)", t)
    if c:
        outs = {base(x.strip()) for x in c.group(1).split(",")}
        ins = {base(x) for x in re.findall(r"\b((?:buf|arg)\w+)\b", c.group(3))}
        if "out=" in c.group(3):
            o = re.search(r"out=(\w+)", c.group(3)).group(1); outs = {base(o)}; ins -= outs
        nodes.append((ins, outs, c.group(2).split(".")[2] if c.group(2).startswith("torch.ops") else c.group(2))); continue
last_w, last_r = {}, collections.defaultdict(list)
level = []
for i, (r, w, _) in enumerate(nodes):
    deps = [last_w[b] for b in r if b in last_w] + [last_w[b] for b in w if b in last_w] + [j for b in w for j in last_r[b]]
    level.append(1 + max((level[j] for j in deps), default=0))
    for b in r: last_r[b].append(i)
    for b in w: last_w[b] = i; last_r[b] = []
hist = collections.Counter(level)
widths = collections.Counter(hist.values())
print(f"{os.path.basename(path)}: {len(nodes)} kernels, critical path {max(level)} levels, mean width {len(nodes) / max(level):.2f}")
print("levels by width (width: count):", dict(sorted(widths.items())))
print("kernels sitting in levels of width >= 2:", sum(w * c for w, c in widths.items() if w >= 2), "of", len(nodes))
