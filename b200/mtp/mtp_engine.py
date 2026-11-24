"""Compiled MTP speculative decoding for qwen4exp (static caches, CUDA graphs): K chained drafts by the MTP head, one
(K+1)-token verify of the main model, on-device rollback of GDN conv / recurrent states and attention lengths.
  python mtp_engine.py [K] [new_tokens] [--offload gpu-nola:0.25]"""
import argparse, copy, json, time
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.integrations.metal_quantization import MetalExperts
from transformers.masking_utils import create_causal_mask
import transformers.models.qwen4_exp.modeling_qwen4_exp as qm

ap = argparse.ArgumentParser(); ap.add_argument("K", type=int, nargs="?", default=2); ap.add_argument("N", type=int, nargs="?", default=128)
ap.add_argument("--offload", default=""); ap.add_argument("--profile", action="store_true"); ap.add_argument("--mode", default="reduce-overhead"); ap.add_argument("--decode-linears", action="store_true")
args = ap.parse_args(); K, N, T = args.K, args.N, args.K + 1
torch._inductor.config.graph_partition = True
torch._inductor.config.combo_kernels = True
path = snapshot_download("ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit", allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*", "*.txt"])
tok = AutoTokenizer.from_pretrained(path)
kw = dict(dtype=torch.bfloat16, experts_implementation="grouped_mm")
if args.offload:
    from transformers import AutoConfig
    from transformers.integrations.cuda_expert_offload import apply_gpu_experts_offload
    n = AutoConfig.from_pretrained(path).text_config.num_hidden_layers
    model = AutoModelForCausalLM.from_pretrained(path, device_map={"": "cuda", **{f"model.layers.{i}.mlp.experts": "cpu" for i in range(n)}}, **kw)
    kind, frac = args.offload.split(":")
    import os
    cpus = None
    if "host" in kind.split("-"):
        lo, hi = map(int, os.environ.get("HOST_TIER_CPUS", "8-47").split("-"))
        cpus = list(range(lo, hi + 1))
    apply_gpu_experts_offload(model, float(frac), path, lookahead=False, host_cpus=cpus)
else:
    model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda", **kw)
if args.decode_linears:
    from transformers.integrations.cuda_decode_linear import kernelize_decode_linears
    print(kernelize_decode_linears(model))
cfg = model.config.get_text_config(); base = model.model; H, HC = cfg.hidden_size, cfg.hc_count
LIN = [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]
ATT = [i for i, t in enumerate(cfg.layer_types) if t != "linear_attention"]

# ---- static side buffers written inside the compiled verify
R_BUF = torch.zeros(1, T, HC * H, device="cuda", dtype=torch.bfloat16)
torch._dynamo.mark_static_address(R_BUF)
STATE_BUF = {}  # GDN layer -> [T, *state] (allocated after prefill)
SPEC = {"on": False}
_orig_rule = qm.torch_chunk_gated_delta_rule
def chunk_rule_with_states(query, key, value, g, beta, initial_state=None, output_final_state=False, **kw):
    if not SPEC["on"] or initial_state is None:
        return _orig_rule(query, key, value, g=g, beta=beta, initial_state=initial_state, output_final_state=output_final_state, **kw)
    outs, state = [], initial_state
    buf = STATE_BUF[SPEC["layer"]]
    for t in range(query.shape[1]):
        o, state = qm.torch_recurrent_gated_delta_rule(query[:, t:t + 1], key[:, t:t + 1], value[:, t:t + 1], g=g[:, t:t + 1],
                                                       beta=beta[:, t:t + 1], initial_state=state, output_final_state=True, **kw)
        outs.append(o); buf[t].copy_(state)
    return torch.cat(outs, 1), state
qm.torch_chunk_gated_delta_rule = chunk_rule_with_states
for i in LIN:
    base.layers[i].linear_attn.register_forward_pre_hook(lambda m, a, i=i: SPEC.__setitem__("layer", i))
def capture_R(m, a):
    if SPEC["on"]:
        R_BUF[:, : a[0].shape[1]].copy_(a[0])
base.hyper_connection_mixer.register_forward_pre_hook(capture_R)

# ---- MTP head
mcfg = copy.deepcopy(cfg)
mcfg.layer_types, mcfg.num_hidden_layers, mcfg.ple_layer_ids = [ATT and cfg.layer_types[ATT[0]]], 1, []
class MtpHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_fc_norm_hidden = qm.Qwen4ExpTextRMSNorm(HC * H, group_size=H, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_embedding = qm.Qwen4ExpTextRMSNorm(H, eps=cfg.rms_norm_eps)
        self.fc_hidden = torch.nn.Linear(H, H, bias=False); self.fc_embedding = torch.nn.Linear(H, H, bias=False)
        self.layers = torch.nn.ModuleList([qm.Qwen4ExpTextDecoderLayer(mcfg, 0)])
        self.hyper_connection_mixer = qm.Qwen4ExpTextGatedResidual(mcfg, use_combine=False)
mtp = MtpHead(); mtp.layers[0].mlp.experts = MetalExperts(mtp.layers[0].mlp.experts, bits=4, group_size=64)
idx = json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]; sd = {}
for k_, f in idx.items():
    if k_.startswith("mtp."):
        with safe_open(f"{path}/{f}", "pt") as fh: sd[k_[4:]] = fh.get_tensor(k_)
mtp.load_state_dict(sd, strict=True); mtp = mtp.to("cuda", torch.bfloat16)
for name in ("gate_up_proj", "down_proj"):
    setattr(mtp.layers[0].mlp.experts, name, torch.nn.Parameter(sd[f"layers.0.mlp.experts.{name}"].cuda(), requires_grad=False))
if args.decode_linears:
    kernelize_decode_linears(mtp)

def mtp_rows(R, next_ids, mcache):
    """MTP rows (R [1, t, HC*H], next ids [1, t]) appended to the MTP cache at its current length -> (last row's logits,
    last row's output residual)."""
    t = R.shape[1]
    h2 = mtp.fc_hidden(mtp.pre_fc_norm_hidden(R).unflatten(-1, (HC, H)))
    e2 = mtp.fc_embedding(mtp.pre_fc_norm_embedding(base.embed_tokens(next_ids)))
    Rm = (h2 + e2.unsqueeze(-2)).flatten(-2)
    pos = (torch.arange(t, device=R.device) + mcache.get_seq_length())[None]
    pe = base.rotary_emb(Rm, pos[None].expand(3, 1, -1))
    mask = create_causal_mask(config=mcfg, inputs_embeds=Rm[..., :H], attention_mask=None, past_key_values=mcache,
                              position_ids=pos, allow_is_causal_skip=False)
    out = mtp.layers[0](Rm, position_embeddings=pe, attention_mask=mask, past_key_values=mcache)
    return model.lm_head(mtp.hyper_connection_mixer(out[:, -1:])), out[:, -1:]

MAX_LEN = 1024

def mtp_rows_variant():
    """A distinct function object (own dynamo cache entry) per row count."""
    def rows(R, next_ids, mcache):
        return mtp_rows(R, next_ids, mcache)
    rows.__code__ = rows.__code__.replace()  # a fresh code object
    return rows

def verify(inp, cache):
    return model(inp, past_key_values=cache, use_cache=True).logits

def sub_len(L, n):
    """Rewind an attention cache layer by `n` positions (K/V, indexer keys, indexer positions)."""
    L.cumulative_length.sub_(n)
    for name in ("indexer_cumulative_length", "indexer_position_length"):
        if getattr(L, name, None) is not None:
            getattr(L, name).sub_(n)

SNAP = {}  # (layer, state_idx) -> pre-verify conv state copy
def snapshot(cache):
    for (li, si), buf in SNAP.items():
        buf.copy_(cache.layers[li].conv_states[si])

def make_rollback(drop):
    a = K - drop
    def rollback(cache, mcache):
        for li in ATT:
            sub_len(cache.layers[li], drop)
        for (li, si), snap in SNAP.items():
            st = cache.layers[li].conv_states[si]
            k = st.shape[-1]
            full = torch.cat([snap, st[..., -T:]], -1)  # old k + the T new columns
            L_ = full.shape[-1]
            st.copy_(full[..., L_ - k - drop: L_ - drop])
        for li in LIN:
            cache.layers[li].recurrent_states[0].copy_(STATE_BUF[li][a])
    return rollback

@torch.inference_mode()
def generate(ids, n_new, compiled):
    P = ids.shape[1]
    cache = StaticCache(config=model.config, max_cache_len=MAX_LEN)  # one size for every call: no shape-driven recompiles
    mcache = StaticCache(config=mcfg, max_cache_len=MAX_LEN)
    SPEC["on"] = False
    hook_R = []
    h = base.hyper_connection_mixer.register_forward_pre_hook(lambda m, a: hook_R.append(a[0]))
    logits = model(ids, past_key_values=cache, use_cache=True).logits
    h.remove()
    x = logits[:, -1:].argmax(-1)
    if not SNAP:
        for li in LIN:
            Lyr = cache.layers[li]
            for si in range(Lyr.number_of_states):
                if Lyr.is_conv_states_initialized[si]:
                    SNAP[(li, si)] = torch.empty_like(Lyr.conv_states[si]); torch._dynamo.mark_static_address(SNAP[(li, si)])
            STATE_BUF[li] = torch.empty((T, *cache.layers[li].recurrent_states[0].shape), device="cuda", dtype=cache.layers[li].recurrent_states[0].dtype)
            torch._dynamo.mark_static_address(STATE_BUF[li])
    fv, fm, fr = compiled
    mlogits, last = mtp_rows(hook_R[-1], torch.cat([ids[:, 1:], x], 1), mcache)  # prompt catch-up (eager, once)
    emitted, steps, acc = [x.item()], 0, 0
    PH = {k: 0.0 for k in ("drafts", "verify", "sync", "rollback", "catchup")}
    ev = lambda: torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    while len(emitted) < n_new:
        e0, e1, e2, e3, e4 = ev(), ev(), ev(), ev(), ev(); e0.record()
        drafts = [mlogits.argmax(-1)]
        mlen = int(mcache.get_seq_length())
        for j in range(1, K):
            ml, last = (t.clone() for t in fm[1](last, drafts[-1], mcache))
            drafts.append(ml.argmax(-1))
        if K > 1:
            sub_len(mcache.layers[0], K - 1)  # chained draft rows are speculative
        e1.record()
        inp = torch.cat([x] + drafts, 1)
        snapshot(cache)
        SPEC["on"] = True
        g = fv(inp, cache).argmax(-1).clone()
        SPEC["on"] = False
        e2.record()
        d = torch.cat(drafts, 1)
        a = int((d == g[:, :K]).int().cumprod(-1).sum())
        ts = time.perf_counter()
        new = g[0, : a + 1].tolist()
        emitted += new; steps += 1; acc += a
        PH["sync"] += time.perf_counter() - ts
        e3.record()
        fr[K - a](cache, mcache)
        mlogits, last = (t.clone() for t in fm[a + 1](R_BUF[:, : a + 1].clone(), g[:, : a + 1].contiguous(), mcache))
        x = g[:, a: a + 1]
        e4.record(); e4.synchronize()
        PH["drafts"] += e0.elapsed_time(e1) / 1e3; PH["verify"] += e1.elapsed_time(e2) / 1e3
        PH["rollback"] += e3.elapsed_time(e4) / 1e3
    torch.cuda.synchronize()
    generate.phases = {k: v / max(steps, 1) * 1e3 for k, v in PH.items()}
    return emitted[:n_new], steps, acc, (len(emitted) - 1) / (time.perf_counter() - t0)

eager = (verify, {t: mtp_rows for t in range(1, T + 1)}, {d: make_rollback(d) for d in range(T)})
if args.mode == "max-autotune":  # the autotuning, with reduce-overhead's cudagraph trees
    torch._inductor.config.max_autotune = True
    torch._inductor.config.coordinate_descent_tuning = True
opts = {"mode": "reduce-overhead", "fullgraph": False, "dynamic": False}
torch._dynamo.config.cache_size_limit = 64
comp = (torch.compile(verify, **opts), {t: torch.compile(mtp_rows_variant(), **opts) for t in range(1, T + 1)},
        {d: torch.compile(make_rollback(d), **opts) for d in range(T)})
PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
           "Explain the causes of the French Revolution to a high school student.",
           "Translate into French and explain the grammar: 'The children would have gone if it had not rained.'"]
with torch.inference_mode():
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).input_ids.cuda()
        ref, *_ = generate(ids, N, eager)
        for _ in range(2): generate(ids, 64, comp)  # warm / capture
        if args.profile:
            from torch.profiler import profile, ProfilerActivity
            import collections
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                _, psteps, _, _ = generate(ids, N, comp)
            ks = sorted((e for e in prof.key_averages() if e.device_time_total > 0), key=lambda e: -e.device_time_total)
            tot = sum(e.device_time_total for e in ks)
            print(f"PROFILE {tot / 1e3 / psteps:.2f} ms GPU per step, {sum(e.count for e in ks) / psteps:.0f} kernels/step")
            for e in ks[:25]:
                print(f"PROFILE {e.device_time_total / 1e3 / psteps:7.3f} ms/step {e.count / psteps:6.1f}/step  {e.key[:110]}")
            args.profile = False
        got, steps, acc, tps = generate(ids, N, comp)
        same = next((i for i in range(N) if got[i] != ref[i]), N)
        print("   per step ms:", {k: round(v, 2) for k, v in generate.phases.items()}, "(rollback includes the catch-up)")
        print(f"K={K} compiled: {tps:.1f} tok/s, {N / steps:.2f} tokens/step, acceptance {acc / (steps * K):.3f}, identical to eager spec for {same}/{N}", flush=True)
