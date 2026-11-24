"""Short decode benchmark: load a streamed-experts checkpoint, compile, time one greedy generation.

Used for the E1-E3 sweeps. Prints the generated text (so garbage is visible) and the store stats.
"""
import glob, os, sys, time
import torch
from dense_quant import quantize_dense
from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig
from huggingface_hub import snapshot_download
from safetensors import safe_open

repo = sys.argv[1]
cache_gib = float(sys.argv[2])
new_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 512
prompt_idx = int(sys.argv[4]) if len(sys.argv) > 4 else 2
tok = AutoTokenizer.from_pretrained(repo)
path = repo if os.path.isdir(repo) else snapshot_download(repo)
resident = 0
for file in glob.glob(f"{path}/*.safetensors"):
    with safe_open(file, "pt") as f:
        for key in f.keys():
            if ".experts." not in key and ".ngram_embedding." not in key:
                meta = f.get_tensor_meta(key) if hasattr(f, "get_tensor_meta") else None
                shape = meta.shape if meta else f.get_slice(key).get_shape()
                resident += torch.Size(shape).numel() * 2
_attn = os.environ.get("ATTN_IMPLEMENTATION")
_model_kwargs = {"attn_implementation": _attn} if _attn else {}
print(f"attn_implementation={_attn}", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, dtype=torch.bfloat16,
    device_map={"": "mps", "experts": "disk", "ngram_embedding": "disk"},
    max_memory={"mps": int(resident + cache_gib * 2**30)},
    **_model_kwargs,
)
quantize_dense(model, "4g32c")
compile_config = CompileConfig()
compile_config._compile_all_devices = True
compiled = {"cache_implementation": "static", "compile_config": compile_config}
prompts = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "Explain the causes of the French Revolution to a high school student.",
    "Translate into French and explain the grammar: 'The children would have gone if it had not rained.'",
]
prompt = prompts[prompt_idx % len(prompts)]
ids = tok.apply_chat_template(
    [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True,
    enable_thinking=False,
).input_ids.to("mps")


class Stamps:
    def __init__(self):
        self.t = []
    def put(self, _):
        self.t.append(time.perf_counter())
    def end(self):
        pass


def generate(n, **kwargs):
    stamps = Stamps()
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False, streamer=stamps, **kwargs)
    steps = [b - a for a, b in zip(stamps.t[2:], stamps.t[3:])]
    return out[0, -n:].tolist(), steps


print("warmup", flush=True)
generate(64, **compiled)
store = model._experts_offload
store.reset_stats()
print("measure", flush=True)
tokens, steps = generate(new_tokens, **compiled)
text = tok.decode(tokens)[:160]
tail = steps[len(steps) // 2 :]
print(f"decode {len(steps) / sum(steps):.2f} tok/s over {len(steps)} steps", flush=True)
print(f"sustained {len(tail) / sum(tail):.2f} tok/s over last {len(tail)} steps", flush=True)
print(f"text {text!r}", flush=True)
print(f"store {store.stats()}", flush=True)
router = getattr(store, "mps_router", None)
if router is not None:
    print(f"router spin={router.spin_seconds:.2f}s work={router.work_seconds:.2f}s "
          f"ahead={router.ahead_reads} jobs={router.jobs_done}", flush=True)
