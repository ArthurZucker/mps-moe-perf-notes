"""Same GPU-timestamp probe for any mode: a stamp after every layer's experts (MODE=full | host). Per token: layer 0 -> 47
(in-graph) and layer 47 -> next token's layer 0 (rest of 47, lm head, sampling, host loop, launch, layer 0 up to MoE)."""
import ctypes as _ct; _ct.CDLL(None).prctl(4, 0, 0, 0, 0)
import os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, CompileConfig
from transformers.integrations.cuda_decode_linear import kernelize_decode_linears
from transformers.integrations.hub_kernels import lazy_load_kernel
from huggingface_hub import snapshot_download
MODE = os.environ.get("MODE", "full")
path = snapshot_download("ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit", allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*", "*.txt"])
tok = AutoTokenizer.from_pretrained(path)
nl = AutoConfig.from_pretrained(path).text_config.num_hidden_layers
PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
           "Explain the causes of the French Revolution to a high school student."]
kernel = lazy_load_kernel("moe-affine")
STAMPS = torch.zeros(1 << 20, dtype=torch.long, device="cuda"); SLOT = torch.zeros(1, dtype=torch.long, device="cuda")
@torch.library.custom_op("probe::stamp", mutates_args=())
def stamp(x: torch.Tensor) -> torch.Tensor:
    kernel.gtimer(STAMPS, SLOT); return x.clone()
@stamp.register_fake
def _(x): return torch.empty_like(x)
if MODE == "nola":
    from transformers.integrations.cuda_expert_offload import apply_gpu_experts_offload
    model = AutoModelForCausalLM.from_pretrained(path, device_map={"": "cuda", **{f"model.layers.{i}.mlp.experts": "cpu" for i in range(nl)}}, dtype=torch.bfloat16)
    apply_gpu_experts_offload(model, 0.25, path, lookahead=False)
elif MODE == "host":
    from transformers.integrations.cuda_expert_offload import apply_gpu_experts_offload
    model = AutoModelForCausalLM.from_pretrained(path, device_map={"": "cuda", **{f"model.layers.{i}.mlp.experts": "cpu" for i in range(nl)}}, dtype=torch.bfloat16)
    cpus = [c for r in os.environ["HOST_TIER_CPUS"].split(",") for c in range(int(r.split("-")[0]), int(r.split("-")[1]) + 1)]
    node_of = {}
    import glob
    for d in glob.glob("/sys/devices/system/node/node*/cpulist"):
        n = int(d.split("/node")[-1].split("/")[0])
        for part in open(d).read().strip().split(","):
            a, _, b = part.partition("-"); node_of.update({c: n for c in range(int(a), int(b or a) + 1)})
    os.environ["HOST_CPU_NODES"] = ",".join(str(node_of[c]) for c in cpus)
    apply_gpu_experts_offload(model, float(os.environ.get("FRAC", "0.25")), path, lookahead=False, host_cpus=cpus)
else:
    model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda", dtype=torch.bfloat16)
if os.environ.get("STRIP_HOOKS") == "1":  # accelerate's dispatch hooks (cpu offload device_map): everything is on cuda now
    from accelerate.hooks import remove_hook_from_module
    n_hooks = sum(1 for m in model.modules() if hasattr(m, "_hf_hook"))
    remove_hook_from_module(model, recurse=True)
    print("removed accelerate hooks from", n_hooks, "modules")
from transformers.integrations.metal_quantization import MetalExperts
for m in model.modules():
    if isinstance(m, MetalExperts):
        f = m.forward
        m.forward = (lambda f: lambda h, i, w: stamp(f(h, i, w)))(f)
# extra stamp: layer 0's router input (before its MoE)
layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
r0 = layers[0].mlp.gate
f0 = r0.forward
r0.forward = lambda h: f0(stamp(h))
PER = nl + 1  # stamps per token
torch._inductor.config.graph_partition = True
torch._inductor.config.combo_kernels = True
kernelize_decode_linears(model)
gk = dict(cache_implementation="static", compile_config=CompileConfig(mode="max-autotune"), do_sample=False)
N = 128
with torch.inference_mode():
    enc = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).input_ids.cuda()
    model.generate(enc(PROMPTS[0]), max_new_tokens=N, min_new_tokens=N, **gk)  # compile + warm
    for p in PROMPTS[1:]:
        torch.cuda.synchronize(); SLOT.zero_()
        st0 = getattr(model, "_gpu_experts_offload", None)
        if st0 is not None and st0.host_tier is not None:
            for tier in (st0.host_tier, getattr(st0, "host_tier_b", None)):
                if tier is not None: st0.kernel.host_tier2_timing(tier[0])
        model.generate(enc(p), max_new_tokens=N, min_new_tokens=N, **gk)
        torch.cuda.synchronize()
        st = STAMPS[: int(SLOT.item())].cpu().tolist()
        st = st[len(st) % PER:]  # whole tokens: [router0, experts0 .. experts47]
        toks = len(st) // PER
        T = lambda t, i: st[t * PER + i]
        ing = sorted(T(t, nl) - T(t, 1) for t in range(1, toks))
        moe0 = sorted(T(t, 1) - T(t, 0) for t in range(1, toks))
        gap = sorted(T(t + 1, 0) - T(t, nl) for t in range(1, toks - 1))
        tot = sorted(T(t + 1, 0) - T(t, 0) for t in range(1, toks - 1))
        m = lambda v: v[len(v) // 2] / 1e6
        store = getattr(model, "_gpu_experts_offload", None)
        if store is not None and store.host_tier is not None:
            for name, tier in (("A", store.host_tier), ("B", getattr(store, "host_tier_b", None))):
                if tier is None: continue
                t = store.kernel.host_tier2_timing(tier[0]); c = max(t[-1], 1)
                print(f"  tier {name}: {t[-1] / max(toks, 1):.1f} calls/token, {t[6] / c:.2f} routes/call, cpu per call gate {t[0] / c / 1e3:.1f} act {t[1] / c / 1e3:.1f} down {t[2] / c / 1e3:.1f} us, slices re-run {t[7]}")
        print(f"{MODE} {p[:30]!r}: per token {m(tot):.2f} ms = layer0 MoE {m(moe0):.2f} + layers 1..47 {m(ing):.2f} + layer47 end -> next layer0 router {m(gap):.2f} ms")
