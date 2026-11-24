"""Eager decode over the bench's 4 prompts: misses/token with and without lookahead, and guess precision."""
import sys, torch, collections
sys.argv = ["x"]
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.integrations.cuda_expert_offload import apply_gpu_experts_offload
from huggingface_hub import snapshot_download
path = snapshot_download("ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit", allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*", "*.txt"])
tok = AutoTokenizer.from_pretrained(path)
n = AutoConfig.from_pretrained(path).text_config.num_hidden_layers
PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
           "Explain the causes of the French Revolution to a high school student.",
           "Translate into French and explain the grammar: 'The children would have gone if it had not rained.'"]
N = 64
for lookahead in (bool(int(__import__("os").environ.get("LA", "1"))),):
    model = AutoModelForCausalLM.from_pretrained(path, device_map={"": "cuda", **{f"model.layers.{i}.mlp.experts": "cpu" for i in range(n)}}, dtype=torch.bfloat16)
    store = apply_gpu_experts_offload(model, 0.25, path, lookahead=lookahead, prefetch_k=int(__import__("os").environ.get("PK", "10")), seed=bool(int(__import__("os").environ.get("SEED", "0"))))
    actual = collections.defaultdict(list)
    orig = type(store)  # wrap plan to record actual ids per layer
    import transformers.integrations.cuda_expert_offload as m
    real_plan = m.experts_plan
    def rec_plan(store_id, ids, layer, *state):
        actual[layer].append(ids.clone()); return real_plan(store_id, ids, layer, *state)
    m.experts_plan = rec_plan
    store.debug_stats = {"misses": 0, "routes": 0, "layer_calls": 0, "prefetched": 0, "guess_sets": []}
    toks = 0
    with torch.inference_mode():
        for p in PROMPTS:
            ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).input_ids.cuda()
            model.generate(ids, max_new_tokens=N, min_new_tokens=N, do_sample=False); toks += N
    m.experts_plan = real_plan
    st = store.debug_stats
    line = f"lookahead={lookahead}: misses/token {float(st['misses'])/toks:.1f}  (routes/token {st['routes']/toks:.0f})"
    if lookahead and st["guess_sets"]:
        # precision: guessed experts later routed in that layer's next call
        cursor = collections.Counter(); hit = tot = 0; rank_hit = [0] * 10; rank_n = [0] * 10
        for layer, g in st["guess_sets"]:
            i = cursor[layer]; cursor[layer] += 1
            if i < len(actual[layer]):
                a = set(actual[layer][i].tolist()); gl = g.tolist()
                hit += len(a & set(gl)); tot += len(gl)
                for r, e in enumerate(gl): rank_hit[r] += e in a; rank_n[r] += 1
        line += f"  prefetched/token {float(st['prefetched'])/toks:.1f}  guess precision {hit/max(tot,1):.2f}"
        line += "\n  precision by guess rank: " + " ".join(f"{h/max(n,1):.2f}" for h, n in zip(rank_hit, rank_n))
    print(line, flush=True)
    del model, store; torch.cuda.empty_cache()
