"""In-context GPU time per decode component: a timed command-buffer commit before each sub-block of every decoder layer
splits the GPU timeline into labelled segments (GPUStartTime / GPUEndTime per command buffer), at ~1 us per split.
Segment k (between commits k-1 and k) is labelled with commit k-1's tag. Totals are per token.

    PYTHONPATH=src:benchmark_v2/inkling2_metal:benchmark_v2/inkling2_metal/hillclimb python hc_components.py configs/<name>.json [tokens]
"""
import collections, json, os, sys, time
import torch
import hc_bench
from hc_bench import gpu_busy, on_ac
if not on_ac() or gpu_busy() > 15:
    sys.exit(f"refusing to run: battery or busy GPU ({gpu_busy()}% before the run, e.g. mediaanalysisd / a browser)")
from torch.utils.cpp_extension import load as load_ext

COMPONENTS = ["pre_attn", "qkv_proj", "attn_inputs", "attention", "o_proj", "post_attn_conv", "pre_mlp", "router", "experts", "post_mlp_conv"]


def install(model):
    here = os.path.expanduser("~/Work/hub-kernels/mps-events")
    build = os.path.join(here, ".build", f"py{sys.version_info[0]}{sys.version_info[1]}")
    ext = load_ext("_mps_events", [os.path.join(here, "mps_events.mm")], build_directory=build,
                   extra_cflags=["-std=c++20", "-O2", "-D_DARWIN_C_SOURCE", "-include", "sys/types.h"],
                   extra_ldflags=["-framework", "Metal", "-framework", "Foundation", "-framework", "QuartzCore"], verbose=False)  # fmt: skip
    ops = torch.library.Library("hc_seg", "DEF")
    ops.define("commit(Tensor(a!) x, int tag) -> ()")
    ops.impl("commit", lambda x, tag: ext.flush_timed(tag), "MPS")
    torch.library.register_fake("hc_seg::commit")(lambda x, tag: None)
    install.keep = (ops, ext)
    layers = [m for n, m in model.named_modules() if n.endswith("language_model.layers")][0]

    def first_tensor(args, kwargs):
        for v in list(args) + list(kwargs.values()):
            if isinstance(v, torch.Tensor):
                return v

    def hook_for(tag):
        def hook(module, args, kwargs):
            torch.ops.hc_seg.commit(first_tensor(args, kwargs), tag)
        return hook

    for li, layer in enumerate(layers):
        targets = {"pre_attn": layer, "qkv_proj": layer.self_attn, "attn_inputs": layer.self_attn.relative_position_bias.linear,
                   "attention": layer.self_attn.sink_proj,
                   "o_proj": layer.self_attn.o_proj,
                   "post_attn_conv": layer.post_attention_convolution, "pre_mlp": layer.mlp_input, "router": layer.mlp,
                   "experts": layer.mlp.experts, "post_mlp_conv": layer.post_mlp_convolution}
        wanted = COMPONENTS if os.environ.get("HC_SEG", "component") == "component" else ["pre_attn"]  # layer: one split per layer
        for ci, name in enumerate(COMPONENTS):
            if name in wanted:
                targets[name].register_forward_pre_hook(hook_for(li * 16 + ci), with_kwargs=True)
    return ext


def main(cfg_path, tokens):
    cfg = json.load(open(cfg_path))
    model, tok, ids, gen = hc_bench.load(cfg)
    ext = install(model)
    stamps = []

    class S:
        def put(self, _): stamps.append(time.perf_counter())
        def end(self): pass

    with torch.inference_mode():
        for _ in range(2):
            model.generate(ids, max_new_tokens=16, min_new_tokens=16, **gen)
        torch.mps.synchronize(); ext.timings()
        t_start = time.perf_counter()
        streamer = None if os.environ.get("HC_NO_STREAMER") == "1" else S()
        model.generate(ids, max_new_tokens=tokens, min_new_tokens=tokens, streamer=streamer, **gen)
        torch.mps.synchronize(); t_total = time.perf_counter() - t_start; time.sleep(0.2)
    t = sorted(ext.timings(), key=lambda r: r[2])  # commit order
    t = t[len(t) // 8:]
    if stamps:
        steps = sorted(b - a for a, b in zip(stamps[2:], stamps[3:]))
        step = steps[len(steps) // 2]
    else:  # no streamer: mean step over the run (prefill included, so a slight overestimate)
        step = t_total / tokens
    span = max(r[1] for r in t) - min(r[0] for r in t)
    n_tok = span / step
    per_comp, per_layer, idle = collections.defaultdict(float), collections.defaultdict(float), 0.0
    idle_after, gap_list = collections.defaultdict(float), []
    prev_tag, prev_end = None, None
    for s0, e0, _, tag in t:
        if prev_tag is not None and prev_tag >= 0:
            per_comp[COMPONENTS[prev_tag % 16]] += e0 - s0
            per_layer[prev_tag // 16] += e0 - s0
        elif prev_tag is not None:
            per_comp["other (lm_head, sampling, ...)"] += e0 - s0
        if prev_end is not None and s0 > prev_end:
            idle += s0 - prev_end
            where = "token boundary (lm_head / sampling / stop check)" if prev_tag is None or prev_tag < 0 else f"after {COMPONENTS[prev_tag % 16]} (layer {prev_tag // 16})"
            idle_after[where.split(" (layer")[0]] += s0 - prev_end
            gap_list.append((s0 - prev_end, where))
        prev_tag, prev_end = tag, max(prev_end or 0, e0)
    total = sum(per_comp.values())
    print(f"== {cfg['name']} [{os.environ.get('HC_SEG', 'component')}{', no streamer' if not stamps else ''}]: host step {step * 1e3:.2f} ms | GPU in segments {total / n_tok * 1e3:.2f} ms/token | "
          f"GPU idle between segments {idle / n_tok * 1e3:.2f} ms/token | {len(t) / n_tok:.0f} segments/token")
    for name, v in sorted(per_comp.items(), key=lambda kv: -kv[1]):
        print(f"   {name:32s} {v / n_tok * 1e3:6.2f} ms/token  ({100 * v / total:4.1f}%)")
    print("   GPU idle by what precedes it (ms/token):", {k: round(v / n_tok * 1e3, 3) for k, v in sorted(idle_after.items(), key=lambda kv: -kv[1])[:6]})
    print("   largest gaps (us):", [(round(g * 1e6), w) for g, w in sorted(gap_list, reverse=True)[:6]])
    print("   per layer (ms/token, loop layers 6-17 run 3x):", {k: round(v / n_tok * 1e3, 2) for k, v in sorted(per_layer.items())})
    json.dump({"name": cfg["name"], "step_ms": step * 1e3, "per_component_ms": {k: v / n_tok * 1e3 for k, v in per_comp.items()},
               "per_layer_ms": {k: v / n_tok * 1e3 for k, v in per_layer.items()}},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles", f"components-{cfg['name']}-{os.environ.get('HC_SEG', 'component')}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 96)
