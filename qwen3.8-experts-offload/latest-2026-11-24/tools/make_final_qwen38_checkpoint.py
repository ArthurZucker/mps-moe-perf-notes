"""Make a final ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit checkpoint: the always-on nn.Linears stored
pre-quantized (Metal affine 4-bit, group 32, clipping search: what `quantize_dense(model, "4g32c")` computes at
every load today), so nothing is quantized when the model loads.

    python tools/make_final_qwen38_checkpoint.py <source snapshot dir> <output dir>

Run with PYTHONPATH=<transformers>/src:<transformers>/benchmark_v2/experts_offload (for `dense_quant`). The
expert files and the n-gram-only files are APFS clones of the source (no extra disk); only the files holding dense
layers are rewritten. The quantization plan in config.json gains one Metal 4/32 entry per quantized module family,
in front of the existing `*` (fp8 n-gram) entry.
"""

import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import transformers
from dense_quant import _clipped_affine


BITS, GROUP, ROWS_PER_CHUNK = 4, 32, 16384


def quantize(weight):
    """`_clipped_affine` on MPS in row chunks (groups lie within a row: identical result, bounded memory)."""
    parts = []
    for start in range(0, weight.shape[0], ROWS_PER_CHUNK):
        w = weight[start : start + ROWS_PER_CHUNK].to("mps").float()
        packed, scales, biases = _clipped_affine(w, GROUP, BITS)
        parts.append((packed.cpu(), scales.to(torch.bfloat16).cpu(), biases.to(torch.bfloat16).cpu()))
        del w, packed, scales, biases
        torch.mps.synchronize()
        torch.mps.empty_cache()
    return tuple(torch.cat(p) for p in zip(*parts))


def quantized_modules(source):
    """Checkpoint names of the nn.Linears `quantize_dense` turns into MetalLinear (all of them on this model)."""
    config = transformers.AutoConfig.from_pretrained(source)
    config.quantization_config = None
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(config)
    names = []
    for name, module in model.named_modules():
        if type(module) is torch.nn.Linear and module.in_features % GROUP == 0:
            # the model's `model.layers...` is the checkpoint's `model.language_model.layers...`
            names.append("model.language_model." + name[len("model.") :] if name.startswith("model.") else name)
    return names


def plan_globs(modules, source):
    """Plan patterns for `modules`. Globs are matched against the MODEL's module names as written (only exact names
    are renamed from the checkpoint's), so each starts with `*` past `model.language_model`, like `*.mlp.experts`."""
    globs = sorted({"*" + re.sub(r"\.\d+\.", ".*.", m)[len("model.language_model") :]
                    if m.startswith("model.language_model.") else m for m in modules})
    config = transformers.AutoConfig.from_pretrained(source)
    config.quantization_config = None
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(config)
    names = [n for n, _ in model.named_modules() if n]
    linear = {n for n, m in model.named_modules() if type(m) is torch.nn.Linear}
    for g in globs:  # the plan matches `*{pattern}.*`-style: every hit must be a quantized Linear
        hits = [n for n in names if fnmatch.fnmatch(n, g) or fnmatch.fnmatch(n, f"*.{g}")]
        if not hits or any(h not in linear for h in hits):
            raise SystemExit(f"plan pattern {g} matches {hits[:3]}")
    covered = {n for n in linear if any(fnmatch.fnmatch(n, g) or fnmatch.fnmatch(n, f"*.{g}") for g in globs)}
    if covered != linear:
        raise SystemExit(f"Linears no pattern covers: {sorted(linear - covered)[:3]}")
    return globs


def final_plan(plan, modules, globs):
    """The source plan + one Metal 4/32 entry per module family, before the `*` (fp8 n-gram) fallback. The fallback's
    `modules_to_not_convert` lists every dense module, and the plan folds those into unquantized patterns that come
    BEFORE globs: the quantized ones must leave that list, else they win over the Metal globs and load as bf16."""
    plan = dict(plan)
    star = plan.pop("*", None)
    plan.update(dict.fromkeys(globs, {"quant_method": "metal", "bits": BITS, "group_size": GROUP}))
    if star is not None:
        quantized = set(modules)
        star = {**star, "modules_to_not_convert": [m for m in star.get("modules_to_not_convert") or [] if m not in quantized]}
        plan["*"] = star
    return plan


def main(source, output):
    index = json.load(open(os.path.join(source, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    modules = quantized_modules(source)
    missing = [m for m in modules if f"{m}.weight" not in weight_map]
    if missing:
        raise SystemExit(f"{len(missing)} modules not in the checkpoint, e.g. {missing[:3]}")
    targets = {f"{m}.weight" for m in modules}
    globs = plan_globs(modules, source)
    print(f"{len(modules)} modules to quantize, {len(globs)} plan patterns", flush=True)

    os.makedirs(output, exist_ok=True)
    files = sorted(set(weight_map.values()))
    rewrite = sorted({weight_map[k] for k in targets})
    for name in sorted(os.listdir(source)):
        src, dst = os.path.join(source, name), os.path.join(output, name)
        if name.startswith(".") or name in ("config.json", "model.safetensors.index.json") or name in rewrite:
            continue
        if not os.path.exists(dst):
            if name.endswith(".safetensors"):
                subprocess.run(["cp", "-c", os.path.realpath(src), dst], check=True)  # APFS clone: no data copied
            else:
                shutil.copy(os.path.realpath(src), dst)
    print(f"cloned {len(files) - len(rewrite)} weight files, rewriting {len(rewrite)}", flush=True)

    new_map = {k: f for k, f in weight_map.items() if f not in rewrite}
    start = time.perf_counter()
    for i, name in enumerate(rewrite, 1):
        tensors = {}
        with safe_open(os.path.join(source, name), "pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key in targets:
                    base = key[: -len(".weight")]
                    packed, scales, biases = quantize(tensor)
                    tensors[key], tensors[f"{base}.scales"], tensors[f"{base}.qbiases"] = packed, scales, biases
                else:
                    tensors[key] = tensor
        save_file(tensors, os.path.join(output, name), metadata={"format": "pt"})
        new_map.update(dict.fromkeys(tensors, name))
        print(f"  [{i}/{len(rewrite)}] {name} ({time.perf_counter() - start:.0f}s)", flush=True)

    index["weight_map"] = dict(sorted(new_map.items()))
    index.setdefault("metadata", {})["total_size"] = sum(
        os.path.getsize(os.path.join(output, f)) for f in set(new_map.values())
    )
    json.dump(index, open(os.path.join(output, "model.safetensors.index.json"), "w"), indent=2)

    config = json.load(open(os.path.join(source, "config.json")))
    plan = config["quantization_config"]
    config["quantization_config"] = final_plan(plan, modules, globs)
    json.dump(config, open(os.path.join(output, "config.json"), "w"), indent=2)
    print(f"done: {output}", flush=True)


def fix_plan(source, output):
    """Rewrite only config.json's plan (the tensors already written are unchanged)."""
    config = json.load(open(os.path.join(source, "config.json")))
    modules = quantized_modules(source)
    plan = final_plan(config["quantization_config"], modules, plan_globs(modules, source))
    config["quantization_config"] = plan
    json.dump(config, open(os.path.join(output, "config.json"), "w"), indent=2)
    print("plan rewritten:", [k for k in plan if k != "*"][:4], "...", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--fix-plan":
        fix_plan(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2])
