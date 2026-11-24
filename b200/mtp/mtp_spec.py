"""Eager greedy speculative decoding with the qwen4exp MTP head (k chained drafts), GDN state rollback.
Checks the output against plain greedy generate and reports accepted tokens per verify step."""
import copy, json, sys, time
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.integrations.metal_quantization import MetalExperts
from transformers.masking_utils import create_causal_mask
import transformers.models.qwen4_exp.modeling_qwen4_exp as qm

K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
N = int(sys.argv[2]) if len(sys.argv) > 2 else 96
path = snapshot_download("ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit", allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*", "*.txt"])
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda", dtype=torch.bfloat16)
cfg = model.config.get_text_config(); base = model.model
H, HC = cfg.hidden_size, cfg.hc_count

# ---- GDN: per-token recurrent states during a verify (seq_len > 1 with a cache that has a previous state)
SPEC = {"on": False, "states": {}}
_orig_rule = qm.torch_chunk_gated_delta_rule
def chunk_rule_with_states(query, key, value, g, beta, initial_state=None, output_final_state=False, **kw):
    if not SPEC["on"] or initial_state is None:
        return _orig_rule(query, key, value, g=g, beta=beta, initial_state=initial_state, output_final_state=output_final_state, **kw)
    outs, states, state = [], [], initial_state
    for t in range(query.shape[1]):  # token by token: the state after every token
        o, state = qm.torch_recurrent_gated_delta_rule(query[:, t:t + 1], key[:, t:t + 1], value[:, t:t + 1], g=g[:, t:t + 1],
                                                       beta=beta[:, t:t + 1], initial_state=state, output_final_state=True, **kw)
        outs.append(o); states.append(state)
    SPEC["states"][SPEC["layer"]] = states
    return torch.cat(outs, 1), state
qm.torch_chunk_gated_delta_rule = chunk_rule_with_states
for i, layer in enumerate(base.layers):  # tag the layer index for the hook above
    if layer.layer_type == "linear_attention":
        layer.linear_attn.register_forward_pre_hook(lambda m, a, i=i: SPEC.__setitem__("layer", i))

# ---- MTP head (as in mtp_accept.py)
mcfg = copy.deepcopy(cfg)
mcfg.layer_types, mcfg.num_hidden_layers, mcfg.ple_layer_ids = [next(t for t in cfg.layer_types if t != "linear_attention")], 1, []
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
for k, f in idx.items():
    if k.startswith("mtp."):
        with safe_open(f"{path}/{f}", "pt") as fh: sd[k[4:]] = fh.get_tensor(k)
mtp.load_state_dict(sd, strict=True); mtp = mtp.to("cuda", torch.bfloat16)
for name in ("gate_up_proj", "down_proj"):
    setattr(mtp.layers[0].mlp.experts, name, torch.nn.Parameter(sd[f"layers.0.mlp.experts.{name}"].cuda(), requires_grad=False))

cap = []
base.hyper_connection_mixer.register_forward_pre_hook(lambda m, a: cap.append(a[0]))

def mtp_rows(R, next_ids, pos0, cache):
    """MTP rows for residuals R [1, T, HC*H] and the next tokens' ids [1, T] at positions pos0..: appends their K/V to the
    MTP cache, returns (logits of the last row, the last row's output residual)."""
    T = R.shape[1]
    h2 = mtp.fc_hidden(mtp.pre_fc_norm_hidden(R).unflatten(-1, (HC, H)))
    e2 = mtp.fc_embedding(mtp.pre_fc_norm_embedding(base.embed_tokens(next_ids)))
    Rm = (h2 + e2.unsqueeze(-2)).flatten(-2)
    pos = (torch.arange(T, device="cuda") + pos0)[None]
    pe = base.rotary_emb(Rm, pos[None].expand(3, 1, -1))
    mask = create_causal_mask(config=mcfg, inputs_embeds=Rm[..., :H], attention_mask=torch.ones(1, pos0 + T, device="cuda", dtype=torch.long),
                              past_key_values=cache, position_ids=pos, allow_is_causal_skip=False)
    out = mtp.layers[0](Rm, position_embeddings=pe, attention_mask=mask, past_key_values=cache)
    return model.lm_head(mtp.hyper_connection_mixer(out[:, -1:])), out[:, -1:]

@torch.inference_mode()
def spec_generate(ids, n_new, mode="mtp", ref=None):
    cache = DynamicCache(config=model.config); cache.activate_past_recording()
    mcache = DynamicCache(config=mcfg)
    cap.clear()
    out = model(ids, past_key_values=cache, use_cache=True)
    x = out.logits[:, -1:].argmax(-1)  # next token (emitted)
    R = cap[-1]
    P = ids.shape[1]
    # MTP catch-up over the prompt: rows t pair R_t with x[t+1] (the last row with x)
    nxt = torch.cat([ids[:, 1:], x], 1)
    logits, last = mtp_rows(R, nxt, 0, mcache)
    emitted = [x.item()]; steps = 0; accepted_total = 0
    pos = P  # position of x
    while len(emitted) < n_new:
        # chain K drafts
        drafts, d = [], logits.argmax(-1)
        drafts.append(d)
        if mode == "oracle":  # the true greedy continuation as drafts (always accepted if the verify reproduces it)
            j0 = len(emitted)
            drafts = [torch.tensor([[ref[min(j0 + j, len(ref) - 1)]]], device="cuda") for j in range(K)]
        elif mode == "reject":  # drafts that can never match: every verify keeps one token
            drafts = [torch.tensor([[0]], device="cuda") for _ in range(K)]
        mlen = mcache.get_seq_length()
        for j in range(1, K if mode == "mtp" else 1):
            logits_j, last = mtp_rows(last, d, pos + j, mcache)
            d = logits_j.argmax(-1); drafts.append(d)
        if K > 1:
            mcache.crop(-(mcache.get_seq_length() - mlen))  # the chained draft rows are speculative
        # verify [x, d1..dK]
        inp = torch.cat([x] + drafts, 1)
        SPEC["on"] = True; SPEC["states"] = {}; cap.clear()
        out = model(inp, past_key_values=cache, use_cache=True)
        SPEC["on"] = False
        g = out.logits.argmax(-1)  # g[i] = token after inp[i]
        a = 0
        while a < K and drafts[a].item() == g[0, a].item():
            a += 1
        new = g[0, : a + 1].tolist()
        emitted += new; steps += 1; accepted_total += a
        # rollback: keep inp[0..a] (a + 1 tokens)
        drop = K - a
        cache.crop(-drop)
        for li, states in SPEC["states"].items():
            cache.layers[li].recurrent_states[0].copy_(states[a])
        R = cap[-1][:, : a + 1]
        # MTP catch-up: rows pair R_i (inp[i]) with g[i] (the token after it), i = 0..a
        logits, last = mtp_rows(R, g[:, : a + 1], pos, mcache)
        x = g[:, a: a + 1]; pos += a + 1
    return emitted[:n_new], steps, accepted_total

PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Explain the causes of the French Revolution to a high school student."]
for p in PROMPTS:
    ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).input_ids.cuda()
    with torch.inference_mode():
        ref = model.generate(ids, max_new_tokens=N, min_new_tokens=N, do_sample=False)[0, ids.shape[1]:].tolist()
    for mode in ("reject", "oracle", "mtp"):
        t = time.perf_counter(); got, steps, acc = spec_generate(ids, N, mode, ref); dt = time.perf_counter() - t
        same = next((i for i in range(N) if got[i] != ref[i]), N)
        print(f"[{mode}] K={K}: identical to greedy for {same}/{N} tokens; {N / steps:.2f} tokens/step; eager {N / dt:.1f} tok/s", flush=True)
        if same < N:
            print("   greedy:", repr(tok.decode(ref[max(0, same - 8): same + 12])), "\n   spec  :", repr(tok.decode(got[max(0, same - 8): same + 12])))
