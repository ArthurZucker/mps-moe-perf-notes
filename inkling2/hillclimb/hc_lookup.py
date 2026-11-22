"""Prompt-lookup speculation on a copy-heavy workload (reasoning effort 0: no thinking preamble), 4-bit dense plan:
greedy (generate's compiled decode) vs lookup drafts padded to a fixed K, so the compiled forward only sees two shapes
(T = 1 when nothing matches, T = K + 1 otherwise). tok/s after the first token, accepted drafts per verify step.

    PYTHONPATH=src:benchmark_v2/inkling2_metal:benchmark_v2/inkling2_metal/hillclimb python .../hc_lookup.py [tokens]
"""
import inspect, json, os, sys, time

from hc_bench import gpu_busy, on_ac

HERE = os.path.dirname(os.path.abspath(__file__))
if not on_ac() or gpu_busy() > 20:
    sys.exit("refusing to run: battery or busy GPU")
import torch
import inductor_mps_fixes
from dspark_real import Stamps
from transformers import AutoModelForImageTextToText, AutoTokenizer, CompileConfig, MetalConfig, activations
from transformers.generation import candidate_generator as CG

inductor_mps_fixes.apply()
# verify steps record conv states (size 4 -> 4 + K + 1) and the graph breaks specialize per cache layer: let every variant compile once
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
tokens = int(next((a for a in sys.argv[1:] if a.isdigit()), 384))
repo = "thinkingmachines/Inkling-2-Mini"
metal = MetalConfig(bits=4, group_size=64)
m = AutoModelForImageTextToText.from_pretrained(repo, device_map="mps", dtype=torch.bfloat16,
    quantization_config={"model.visual.*": None, "*_message_gate": None, "lm_head": metal, "*": metal}).eval()
for mod in m.modules():
    if type(mod).__name__.endswith("RelativePositionBias"):
        mod.proj_t = mod.proj.detach().t().to(torch.bfloat16).contiguous()
tok = AutoTokenizer.from_pretrained(repo)
src = inspect.getsource(activations.GELUActivation) + inspect.getsource(activations.FastGELUActivation) + inspect.getsource(activations.QuickGELUActivation) + inspect.getsource(activations.ClippedGELUActivation)
prompt = f"Rewrite this code exactly, only renaming `input` to `hidden_states`. Output only the code, no explanation.\n\n```python\n{src}```"
ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt",
                              return_dict=True, reasoning_effort="none")["input_ids"].to("mps")
print("prompt tokens", ids.shape[1], flush=True)

K = [8]
accepted = []
get_candidates, update = CG.PromptLookupCandidateGenerator.get_candidates, CG.PromptLookupCandidateGenerator.update_candidate_strategy

def padded(self, input_ids, **kw):
    cand, logits = get_candidates(self, input_ids, **kw)
    n = cand.shape[1] - input_ids.shape[1]
    room = self.max_length - 1 - input_ids.shape[1]  # never past the cache (lookup caps its own drafts the same way)
    if 0 < n < min(K[0], room):  # pad to K drafts (repeat the last one): a single verify shape
        cand = torch.cat([cand, cand[:, -1:].expand(-1, min(K[0], room) - n)], dim=1)
    return cand, logits

def record(self, input_ids, scores, num_matches):
    accepted.append(int(num_matches)); return update(self, input_ids, scores, num_matches)

CG.PromptLookupCandidateGenerator.get_candidates = padded
CG.PromptLookupCandidateGenerator.update_candidate_strategy = record
cc = CompileConfig(); cc._compile_all_devices = True
from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList


def budget(n):  # cache with 16 slots of headroom past the last verify step, stopping at exactly n new tokens
    return {"max_new_tokens": n + 16, "stopping_criteria": StoppingCriteriaList([MaxLengthCriteria(ids.shape[1] + n)])}


base = {"do_sample": False, "cache_implementation": "static"}
results = {}
with torch.inference_mode():
    MODES = [("greedy", {"compile_config": cc}), ("lookup-k4", {"k": 4}), ("lookup-k8", {"k": 8})]
    if os.environ.get("HC_ONLY"): MODES = [x for x in MODES if x[0] in os.environ["HC_ONLY"].split(",")]
    for i, (name, kw) in enumerate(MODES):
        if name.startswith("lookup"):
            K[0] = kw.pop("k")
            if not hasattr(m, "_spec_fwd"):
                torch._dynamo.reset()
                eager_forward = type(m).forward.__get__(m)
                compiled = torch.compile(eager_forward, dynamic=False)

                def forward(*a, **k):  # compile decode / verify steps only; prefill stays eager
                    x = k.get("input_ids", a[0] if a else None)
                    return compiled(*a, **k) if x is not None and x.shape[1] <= 16 else eager_forward(*a, **k)

                m._spec_fwd = m.forward = forward
            kw |= {"prompt_lookup_num_tokens": K[0], "disable_compile": True}
        for _ in range(2):
            m.generate(ids, **(base | kw | budget(48)))  # compile / warm both shapes
        rates = []
        for _ in range(3):
            accepted.clear(); st = Stamps()
            out = m.generate(ids, streamer=st, **(base | kw | budget(tokens)))[0, ids.shape[1]:]
            rates.append(st.decode_rate())
        ref = results.get("greedy", {}).get("out")
        same = None if ref is None else int((ref[: len(out)] != out[: len(ref)]).nonzero()[0].item()) if (ref[: len(out)] != out[: len(ref)]).any() else -1
        acc = sum(accepted) / max(len(accepted), 1)
        results[name] = {"rate": sorted(rates)[1], "out": out, "acc": acc}
        print(f"[{i + 1}/3] {name:9s} {sorted(rates)[1]:6.1f} tok/s | accepted/step {acc:.2f} over {len(accepted)} steps | "
              f"first mismatch vs greedy {same} (-1 = identical) | {len(out)} tokens", flush=True)
print(repr(tok.decode(results["greedy"]["out"])[:160]))
with open(os.path.join(HERE, "ledger.jsonl"), "a") as f:
    f.write(json.dumps({"config": "lookup-copy", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "tok_s": {k: round(v["rate"], 2) for k, v in results.items()},
                        "accepted": {k: round(v["acc"], 2) for k, v in results.items()}}) + "\n")
