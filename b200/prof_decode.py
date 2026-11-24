"""Profile compiled (static cache + CUDA graphs) decode of a model: top CUDA kernels per token, graph breaks.

  python prof_decode.py --model ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit [--offload 0.25]
"""
import ctypes as _ct; _ct.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE=0: a crash leaves no huge core (pinned banks)
import argparse, os, sys, time, json
p = argparse.ArgumentParser()
p.add_argument("--model", default="ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit")
p.add_argument("--offload", type=float, default=0.0, help="ours:<fraction> (0 = full residency)")
p.add_argument("--gpu-offload", type=float, default=0.0)
p.add_argument("--pk", type=int, default=0)
p.add_argument("--nola", action="store_true")
p.add_argument("--host", default="", help="host tier cpus a-b (gpu-offload)")
p.add_argument("--tokens", type=int, default=64)
p.add_argument("--eager", action="store_true")
p.add_argument("--gemv", action="store_true")
p.add_argument("--fuse-only", action="store_true")
p.add_argument("--cprofile", action="store_true")
p.add_argument("--mode", default="")
p.add_argument("--inductor", default="", help="k=v,k=v inductor config")
p.add_argument("--no-profile", action="store_true")
p.add_argument("--trace", default="")
p.add_argument("--top", type=int, default=25)
args = p.parse_args()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

path = snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*"])
tok = AutoTokenizer.from_pretrained(path)
kw = dict(dtype=torch.bfloat16, experts_implementation="grouped_mm")
if args.offload:
    from safetensors import safe_open
    import glob
    eb = rb = 0
    for f in glob.glob(f"{path}/*.safetensors"):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                sl = s.get_slice(k); n = torch.Size(sl.get_shape()).numel() * {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "I64": 8}.get(sl.get_dtype(), 1)
                if ".experts." in k: eb += n
                else: rb += n
    model = AutoModelForCausalLM.from_pretrained(path, device_map={"": "cuda", "experts": "disk"}, max_memory={0: int(rb + args.offload * eb)}, **kw)
elif args.gpu_offload:
    from transformers import AutoConfig
    from transformers.integrations.cuda_expert_offload import apply_gpu_experts_offload
    n = AutoConfig.from_pretrained(path).text_config.num_hidden_layers
    model = AutoModelForCausalLM.from_pretrained(path, device_map={"": "cuda", **{f"model.layers.{i}.mlp.experts": "cpu" for i in range(n)}}, **kw)
    print(f"after load: {torch.cuda.memory_allocated()/2**30:.1f} GiB")
    hc = list(range(*[int(c) + i for i, c in enumerate(args.host.split("-"))])) if args.host else None
    store = apply_gpu_experts_offload(model, args.gpu_offload, path, lookahead=not args.nola and not hc, prefetch_k=args.pk or None, host_cpus=hc)
    print(f"after offload setup: {torch.cuda.memory_allocated()/2**30:.1f} GiB")
else:
    model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda", **kw)
torch._inductor.config.graph_partition = True
for kv in filter(None, args.inductor.split(",")):
    k, v = kv.split("="); setattr(torch._inductor.config, k, eval(v)); print("inductor", k, "=", v)
if args.gemv:
    from transformers.integrations.cuda_decode_linear import kernelize_decode_linears
    print("kernelize_decode_linears:", kernelize_decode_linears(model, gemv=not args.fuse_only))
ids = tok.apply_chat_template([{"role": "user", "content": "Write a short story about a lighthouse keeper who finds a message in a bottle."}],
                              add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).input_ids.cuda()
gk = {} if args.eager else {"cache_implementation": "static"}
if args.mode:
    from transformers import CompileConfig
    gk["compile_config"] = CompileConfig(mode=args.mode)
def gen(n):
    return model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False, **gk)
with torch.inference_mode():
    class Stamp:
        def __init__(self): self.t = []
        def put(self, v): torch.cuda.synchronize(); self.t.append(time.perf_counter())
        def end(self): pass
    for _ in range(3):
        gen(args.tokens + 1)
    st = Stamp()
    out = model.generate(ids, max_new_tokens=args.tokens + 1, min_new_tokens=args.tokens + 1, do_sample=False, streamer=st, **gk)
    dt = sorted(b - a for a, b in zip(st.t[2:], st.t[3:]))
    print(f"memory: allocated {torch.cuda.memory_allocated()/2**30:.1f} GiB, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB, reserved {torch.cuda.memory_reserved()/2**30:.1f} GiB")
    med = dt[len(dt) // 2]
    print(f"decode median {med * 1e3:.2f} ms/token -> {1 / med:.1f} tok/s (min {dt[0]*1e3:.2f} max {dt[-1]*1e3:.2f})")
    print("text:", tok.decode(out[0, ids.shape[1]:])[:200].replace("\n", " "))
    if args.host and getattr(store, "host_v2", False):
        k = store.kernel
        k.host_tier2_timing(store.host_tier[0])
        gen(args.tokens + 1); torch.cuda.synchronize()
        t = k.host_tier2_timing(store.host_tier[0])
        c = max(t[-1], 1)
        print(f"host tier: {t[-1] / (args.tokens + 1):.1f} calls/token (of 48 layers), {t[6] / (args.tokens + 1):.1f} host routes/token, cpu per call gate {t[0]/c/1e3:.1f} act {t[1]/c/1e3:.1f} down {t[2]/c/1e3:.1f} us, bad posts {t[3]} bad hints {t[4]} hints {t[5]}")
    if args.cprofile:
        import cProfile, pstats
        pr = cProfile.Profile(); pr.enable(); gen(args.tokens + 1); torch.cuda.synchronize(); pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(25)
        pstats.Stats(pr).sort_stats("cumtime").print_stats(30)
    if not args.no_profile:
        from torch.profiler import profile, ProfilerActivity
        import collections, re
        def kern(n):
            gen(n)  # same compiled shapes; warm
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                gen(n); torch.cuda.synchronize()
            if args.trace:
                prof.export_chrome_trace(args.trace)
            d = collections.defaultdict(lambda: [0.0, 0])
            for e in prof.key_averages():
                if e.device_time_total > 0 and (getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA):
                    d[e.key][0] += e.device_time_total; d[e.key][1] += e.count
            return d
        # same max_new_tokens -> same static cache -> no recompile; prefill-only run = stop after 1 token via max_new_tokens? use eos-free short run
        full = kern(args.tokens + 1)
        def cat(k):
            for pat, c in [("qmv4_kernel<64, 4>", "experts gate_up (cuda)"), ("qmv4_kernel<64, 1>", "experts down_route (cuda)"), ("gather_qmv", "experts triton (prefill)"),
                           ("nvjet|cublas|gemv|splitK|dot_kernel|reduce_1Block|gemm", "cuBLAS dense"), ("fmha|flash|attention", "attention"),
                           ("topk|Topk|TopK|sort|Sort", "topk/sort"), ("conv", "conv"), ("triton", "triton fused"), ("Memcpy|Memset", "memcpy")]:
                if re.search(pat, k): return c
            return "other"
        n = args.tokens + 1
        cats = collections.defaultdict(lambda: [0.0, 0])
        for k, (t, c) in full.items():
            cats[cat(k)][0] += t; cats[cat(k)][1] += c
        tot = sum(v[0] for v in cats.values())
        print(f"GPU kernel time {tot/1e3/n:.2f} ms/token (incl. 1 prefill amortized over {n}), kernels/token {sum(v[1] for v in cats.values())/n:.0f}")
        for c, (t, cnt) in sorted(cats.items(), key=lambda x: -x[1][0]):
            print(f"  {c:28s} {t/1e3/n:7.3f} ms/tok {cnt/n:7.1f}/tok")
        rows = sorted(full.items(), key=lambda x: -x[1][0])
        for k, (t, c) in rows[:args.top if hasattr(args, "top") else 25]:
            print(f"{t/1e3/n:7.3f} ms/tok {c/n:7.1f}/tok  {k[:120]}")
