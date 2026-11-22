"""Speculative decoding for Inkling-2-Mini on MPS (4-bit dense plan, static cache, compile), by workload: greedy vs
prompt-lookup drafts vs the bundled DSpark drafter; tok/s after the first token, greedy exactness, accepted/step.

    PYTHONPATH=src:benchmark_v2/inkling2_metal:benchmark_v2/inkling2_metal/hillclimb python .../hc_spec.py [tokens]
"""
import inspect, json, os, sys, time

from hc_bench import gpu_busy, on_ac

HERE = os.path.dirname(os.path.abspath(__file__))
if not on_ac() or gpu_busy() > 20:
    sys.exit("refusing to run: battery or busy GPU")
import torch
import inductor_mps_fixes
from dspark_real import Stamps, load_drafter
from transformers import AutoModelForImageTextToText, AutoTokenizer, CompileConfig, MetalConfig
from transformers import activations

inductor_mps_fixes.apply()
tokens = int(next((a for a in sys.argv[1:] if a.isdigit()), 256))
repo = "thinkingmachines/Inkling-2-Mini"
metal = MetalConfig(bits=4, group_size=64)
plan = {"model.visual.*": None, "*_message_gate": None, "lm_head": metal, "*": metal}
model = AutoModelForImageTextToText.from_pretrained(repo, device_map="mps", dtype=torch.bfloat16, quantization_config=plan).eval()
for module in model.modules():
    if type(module).__name__.endswith("RelativePositionBias"):
        module.proj_t = module.proj.detach().t().to(torch.bfloat16).contiguous()
drafter = load_drafter(model)
tok = AutoTokenizer.from_pretrained(repo)
source = inspect.getsource(activations.GELUActivation) + inspect.getsource(activations.FastGELUActivation) + inspect.getsource(activations.QuickGELUActivation)
WORKLOADS = {
    "copy-edit": f"Rewrite this code exactly, only renaming `input` to `hidden_states`. Output only the code.\n\n```python\n{source}```",
    "prose": "Write a short story about a lighthouse keeper.",
}
cc = CompileConfig(); cc._compile_all_devices = True
greedy = {"do_sample": False, "max_new_tokens": tokens, "min_new_tokens": tokens, "cache_implementation": "static", "compile_config": cc}
MODES = {"greedy": {}, "lookup": {"prompt_lookup_num_tokens": 10}, "dspark": {"assistant_model": drafter, "speculation_type": "dspark"}}
# assisted decoding records the cache history to roll it back, which generate does not compile: compile the forward as is
ASSISTED = {"disable_compile": True}

def ids_of(p):
    return tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to("mps")

results, n, start = {}, len(WORKLOADS) * len(MODES), time.perf_counter()
with torch.inference_mode():
    i = 0
    for mode, extra in MODES.items():
        kw = greedy | extra
        if mode != "greedy":
            kw |= ASSISTED
            if not hasattr(model, "_spec_forward"):
                model._spec_forward = model.forward = torch.compile(model.forward, dynamic=False)
        model.generate(ids_of("Hi"), **(kw | {"max_new_tokens": 32, "min_new_tokens": 32}))  # compile / warm
        for w, prompt in WORKLOADS.items():
            i += 1
            ids = ids_of(prompt)
            try:
                rates, out = [], None
                for _ in range(3):
                    st = Stamps(); out = model.generate(ids, streamer=st, **kw)[0, ids.shape[1]:]; rates.append(st.decode_rate())
                rate = sorted(rates)[1]
                ref = results.get((w, "greedy"), {}).get("out")
                exact = None if ref is None else bool(torch.equal(ref, out))
                results[(w, mode)] = {"rate": rate, "out": out, "exact": exact}
                print(f"[{i}/{n}] {w:9s} {mode:6s} {rate:6.1f} tok/s  exact_vs_greedy={exact} | elapsed {time.perf_counter() - start:.0f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{n}] {w} {mode}: failed {type(e).__name__}: {str(e)[:160]}", flush=True)
print(repr(tok.decode(results[("copy-edit", "greedy")]["out"])[:200]))
with open(os.path.join(HERE, "ledger.jsonl"), "a") as f:
    f.write(json.dumps({"config": "spec", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "tok_s": {f"{w}/{m}": round(r["rate"], 2) for (w, m), r in results.items()},
                        "exact": {f"{w}/{m}": r["exact"] for (w, m), r in results.items()}}) + "\n")
