"""Compiled decode at 0.25 with the host tier, bench prompts; GPU timestamps (HOST_STAMPS=1) after each layer's resident
experts and after its join: per-layer join stall, and per token: in-graph time (layer 0 -> layer 47 stamps) vs the rest."""
import ctypes as _ct; _ct.CDLL(None).prctl(4, 0, 0, 0, 0)
import os, time, torch
os.environ["HOST_STAMPS"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, CompileConfig
from transformers.integrations.cuda_expert_offload import apply_gpu_experts_offload
from transformers.integrations.cuda_decode_linear import kernelize_decode_linears
from huggingface_hub import snapshot_download
path = snapshot_download("ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit", allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*", "*.txt"])
tok = AutoTokenizer.from_pretrained(path)
nl = AutoConfig.from_pretrained(path).text_config.num_hidden_layers
PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
           "Explain the causes of the French Revolution to a high school student.",
           "Translate into French and explain the grammar: 'The children would have gone if it had not rained.'"]
cpus = os.environ["HOST_TIER_CPUS"].split("-")
model = AutoModelForCausalLM.from_pretrained(path, device_map={"": "cuda", **{f"model.layers.{i}.mlp.experts": "cpu" for i in range(nl)}}, dtype=torch.bfloat16)
store = apply_gpu_experts_offload(model, 0.25, path, lookahead=False, host_cpus=list(range(int(cpus[0]), int(cpus[1]) + 1)))
torch._inductor.config.graph_partition = True
torch._inductor.config.combo_kernels = True
kernelize_decode_linears(model)
gk = dict(cache_implementation="static", compile_config=CompileConfig(mode="max-autotune"), do_sample=False)
N = 128
with torch.inference_mode():
    ids0 = tok.apply_chat_template([{"role": "user", "content": PROMPTS[0]}], add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).input_ids.cuda()
    model.generate(ids0, max_new_tokens=8, min_new_tokens=8, **gk)  # compile
    model.generate(ids0, max_new_tokens=N, min_new_tokens=N, **gk)
    res = []
    for p in PROMPTS[1:]:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).input_ids.cuda()
        torch.cuda.synchronize(); store.stamp_slot.zero_()
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=N, min_new_tokens=N, **gk)
        torch.cuda.synchronize(); wall = time.perf_counter() - t0
        n = int(store.stamp_slot.item()); st = store.stamps[:n].cpu().tolist()
        res.append((p, wall, st))
for p, wall, st in res:
    # decode stamps come in 2 * nl per token (prefill runs the staged path: no stamps)
    per = 2 * nl
    toks = len(st) // per
    stall, ingraph, outside = [], [], []
    for t in range(toks):
        s = st[t * per:(t + 1) * per]
        stall += [s[2 * l + 1] - s[2 * l] for l in range(nl)]
        ingraph.append(s[-1] - s[0])
        if t + 1 < toks: outside.append(st[(t + 1) * per] - s[-1])
    stall.sort(); ingraph.sort(); outside.sort()
    med = lambda v: v[len(v) // 2] / 1e3
    print(f"{p[:40]!r}: wall {wall * 1e3 / N:.2f} ms/token; per token median: layer0->47 in-graph {med(ingraph):.2f} ms, "
          f"layer47 -> next layer0 {med(outside):.2f} ms; join stall sum {sum(stall) / toks / 1e3:.2f} ms/token "
          f"(per layer p50 {med(stall) * 1e3:.1f} us p90 {stall[int(.9 * len(stall))] / 1e3:.1f} us)")
t = store.kernel.host_tier2_timing(store.host_tier[0])
print("cpu per call (us) gate/act/down:", [round(v / max(t[-1], 1) / 1e3, 1) for v in t[:3]], "calls", t[-1], "routes", t[6])
