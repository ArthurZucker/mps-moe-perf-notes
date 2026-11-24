"""Eager decode at 0.25 (LRU pool, misses installed): per layer and token, the missed experts, and how many of them
the guess topk(router_{l+1}(x_l)) (x_l: layer l's MoE input) predicted, for guess sizes 4 / 6 / 10 / 16."""
import ctypes as _ct; _ct.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE=0: a crash leaves no huge core (pinned banks)
import os, sys, json, collections, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.integrations.cuda_expert_offload import apply_gpu_experts_offload
import transformers.integrations.cuda_expert_offload as m
from huggingface_hub import snapshot_download
path = snapshot_download("ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit", allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*", "*.txt"])
tok = AutoTokenizer.from_pretrained(path)
nl = AutoConfig.from_pretrained(path).text_config.num_hidden_layers
PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
           "Explain the causes of the French Revolution to a high school student.",
           "Translate into French and explain the grammar: 'The children would have gone if it had not rained.'"]
N = int(os.environ.get("N", "128"))
FRAC = float(os.environ.get("FRAC", "0.25"))
model = AutoModelForCausalLM.from_pretrained(path, device_map={"": "cuda", **{f"model.layers.{i}.mlp.experts": "cpu" for i in range(nl)}}, dtype=torch.bfloat16)
store = apply_gpu_experts_offload(model, FRAC, path, lookahead=False)
experts = [model.model.language_model.layers[i].mlp.experts if hasattr(model.model, "language_model") else model.model.layers[i].mlp.experts for i in range(nl)]
guesses = {}; guesses2 = {}
SIZES = (4, 6, 10, 16)
layers_mod = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
def pre(layer):
    def hook(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        if h.shape[-2] == 1 if h.dim() == 3 else h.shape[0] == 1:
            with torch.no_grad():
                xm, _, _ = mod.mlp_hyper_connection(h)  # the MoE input without this layer's attention injection
                xm = xm.reshape(-1, xm.shape[-1])
                guesses2[layer] = F.linear(xm, store.routers[layer].weight).float().topk(max(SIZES), dim=-1).indices.reshape(-1).tolist()
    return hook
for l in range(nl): layers_mod[l].register_forward_pre_hook(pre(l), with_kwargs=True)
def wrap(layer, mod):
    orig = mod.forward
    def fwd(h, idx, w):
        if h.shape[0] == 1 and layer + 1 < nl:
            guesses[layer + 1] = F.linear(h, store.routers[layer + 1].weight).float().topk(max(SIZES), dim=-1).indices.reshape(-1).tolist()
        return orig(h, idx, w)
    mod.forward = fwd
for l, e in enumerate(experts): wrap(l, e)
rows = []  # (prompt, step, layer, misses, hinted per size)
real_plan = m.experts_plan
cur = {"p": 0, "t": 0}
def rec_plan(store_id, ids, layer, *state):
    slots, miss = real_plan(store_id, ids, layer, *state)
    if ids.numel() <= 10:
        mi = ids[miss.bool()].tolist()
        g = guesses.get(layer)
        hinted = [len(set(mi) & set(g[:s])) if g else 0 for s in SIZES]
        g2 = guesses2.get(layer)
        hinted2 = [len(set(mi) & set(g2[:s])) if g2 else 0 for s in SIZES]
        rows.append((cur["p"], layer, len(mi), hinted, hinted2))
    return slots, miss
m.experts_plan = rec_plan
with torch.inference_mode():
    for pi, p in enumerate(PROMPTS):
        cur["p"] = pi
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).input_ids.cuda()
        model.generate(ids, max_new_tokens=N, min_new_tokens=N, do_sample=False)
json.dump(rows, open(f"miss_dist2_{FRAC}.json", "w"))
toks = len(rows) / nl
miss = [r[2] for r in rows]
print(f"frac {FRAC}: decode calls {len(rows)} (~{toks:.0f} tokens), misses/token {sum(miss) / toks:.1f}, per layer mean {sum(miss) / len(miss):.2f}")
hist = collections.Counter(miss)
print("misses per layer-call histogram:", {k: hist[k] for k in sorted(hist)})
for i, s in enumerate(SIZES):
    print(f"top-{s}: V1 router(l+1)(x_moe_l) covers {sum(r[3][i] for r in rows) / max(sum(miss), 1):.2f} of misses | "
          f"V2 router(l+1)(mlp_hc(l+1)(streams in)) covers {sum(r[4][i] for r in rows) / max(sum(miss), 1):.2f}")
# after warm-up (skip first 16 tokens of each prompt)
