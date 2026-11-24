"""Record decode routing (layer, token -> top-k experts) for the bench's 4 chat prompts x N tokens (full residency, eager)."""
import sys, torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
path = snapshot_download("ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit", allow_patterns=["*.json", "*.jinja", "tokenizer*", "*.txt"])
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda", dtype=torch.bfloat16)
PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
           "Explain the causes of the French Revolution to a high school student.",
           "Translate into French and explain the grammar: 'The children would have gone if it had not rained.'"]
experts = [m for n, m in model.named_modules() if n.endswith("mlp.experts")]
log = []
def hook(i):
    def f(mod, args, kwargs):
        idx = args[1] if len(args) > 1 else kwargs["top_k_index"]
        log.append((i, idx.detach().cpu()))
    return f
for i, m in enumerate(experts): m.register_forward_pre_hook(hook(i), with_kwargs=True)
out = []
with torch.inference_mode():
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).input_ids.cuda()
        log.clear()
        model.generate(ids, max_new_tokens=N, min_new_tokens=N, do_sample=False)
        # per forward call: 48 entries; first call is the prompt (many tokens)
        steps = [log[i:i + len(experts)] for i in range(0, len(log), len(experts))]
        out.append({"prompt": [[e.tolist() for _, e in steps[0]]], "decode": [[e[0].tolist() for _, e in s] for s in steps[1:]]})
json.dump({"layers": len(experts), "num_experts": 512, "runs": out}, open(f"routing_{N}.json", "w"))
print("saved", [len(r["decode"]) for r in out])
