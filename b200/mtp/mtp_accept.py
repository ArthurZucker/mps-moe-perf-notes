"""Depth-1 MTP acceptance for Qwen3.8-Flash-Next (qwen4_exp), teacher-forced: the MTP head (ref: basalt
src/models/qwen4exp/decode/mtp/forward.cpp) predicts x[t+2] from the main model's pre-mixer residual at t and the
embedding of x[t+1]; compared with the main model's own greedy x[t+2]."""
import copy, json, sys
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.metal_quantization import MetalExperts
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer, Qwen4ExpTextGatedResidual, Qwen4ExpTextRMSNorm,
)

path = snapshot_download("ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit", allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*", "*.txt"])
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda", dtype=torch.bfloat16)
cfg = model.config.get_text_config()
base = model.model
print("main layer types:", sorted(set(cfg.layer_types)))

# --- the MTP head
mcfg = copy.deepcopy(cfg)
attn_type = next(t for t in cfg.layer_types if t != "linear_attention")
mcfg.layer_types, mcfg.num_hidden_layers, mcfg.ple_layer_ids = [attn_type], 1, []
H, HC = cfg.hidden_size, cfg.hc_count
class MtpHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(HC * H, group_size=H, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(H, eps=cfg.rms_norm_eps)
        self.fc_hidden = torch.nn.Linear(H, H, bias=False)
        self.fc_embedding = torch.nn.Linear(H, H, bias=False)
        self.layers = torch.nn.ModuleList([Qwen4ExpTextDecoderLayer(mcfg, 0)])
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(mcfg, use_combine=False)
mtp = MtpHead()
mtp.layers[0].mlp.experts = MetalExperts(mtp.layers[0].mlp.experts, bits=4, group_size=64)
idx = json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]
sd = {}
for k, f in idx.items():
    if k.startswith("mtp."):
        with safe_open(f"{path}/{f}", "pt") as fh: sd[k[len("mtp."):]] = fh.get_tensor(k)
missing, unexpected = mtp.load_state_dict(sd, strict=False)
print("mtp load: missing", missing, "unexpected", unexpected)
mtp = mtp.to("cuda", torch.bfloat16)
for name in ("gate_up_proj", "down_proj"):  # packed codes stay uint32
    setattr(mtp.layers[0].mlp.experts, name, torch.nn.Parameter(sd[f"layers.0.mlp.experts.{name}"].cuda(), requires_grad=False))

# --- main model greedy run, capturing the pre-mixer residual at every position
cap = []
h = base.hyper_connection_mixer.register_forward_pre_hook(lambda m, a: cap.append(a[0].detach()))
PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
           "Explain the causes of the French Revolution to a high school student.",
           "Translate into French and explain the grammar: 'The children would have gone if it had not rained.'"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
tot = {0: [0, 0], 1: [0, 0]}
d2 = [0, 0]  # depth-2: both drafts right (chained) / rows
with torch.inference_mode():
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).input_ids.cuda()
        seq = model.generate(ids, max_new_tokens=N, min_new_tokens=N, do_sample=False)
        cap.clear()
        model(seq)  # one teacher-forced pass over prompt + generation: R for every position
        R = cap[-1]  # [1, T, HC*H]
        T = seq.shape[1]
        x_next = seq[:, 1:]  # embedding input for row t: x[t+1]
        Rin = R[:, : T - 1]
        hn = mtp.pre_fc_norm_hidden(Rin).unflatten(-1, (HC, H))
        h2 = mtp.fc_hidden(hn)
        e2 = mtp.fc_embedding(mtp.pre_fc_norm_embedding(base.embed_tokens(x_next)))
        Rm = (h2 + e2.unsqueeze(-2)).flatten(-2)
        for shift in (0, 1):  # positions of the rows: t, or t + 1 (the embedded token's)
            pos = (torch.arange(T - 1, device="cuda") + shift)[None]
            pos3 = pos[None].expand(3, 1, -1)
            pe = base.rotary_emb(Rm, pos3)
            mask = create_causal_mask(config=mcfg, inputs_embeds=Rm[..., :H], attention_mask=torch.ones_like(pos), past_key_values=None, position_ids=pos, allow_is_causal_skip=False)
            out = mtp.layers[0](Rm, position_embeddings=pe, attention_mask=mask, past_key_values=None)
            logits = model.lm_head(mtp.hyper_connection_mixer(out))
            pred = logits.argmax(-1)[0]  # row t predicts x[t+2]
            target = seq[0, 2:]
            gen0 = ids.shape[1] - 1  # rows whose target is a generated token
            ok = (pred[gen0: T - 2] == target[gen0:]).sum().item()
            tot[shift][0] += ok; tot[shift][1] += target[gen0:].numel()
            if shift == 0:
                # depth 2: the same layer on its own output residual (rows t < T-2) + embedding of x[t+2] -> x[t+3]
                e2b = mtp.fc_embedding(mtp.pre_fc_norm_embedding(base.embed_tokens(seq[:, 2:])))
                R2 = (mtp.fc_hidden(mtp.pre_fc_norm_hidden(out[:, : T - 2]).unflatten(-1, (HC, H))) + e2b.unsqueeze(-2)).flatten(-2)
                pos2 = (torch.arange(T - 2, device="cuda") + 1)[None]
                pe2 = base.rotary_emb(R2, pos2[None].expand(3, 1, -1))
                mask2 = create_causal_mask(config=mcfg, inputs_embeds=R2[..., :H], attention_mask=torch.ones_like(pos2), past_key_values=None, position_ids=pos2, allow_is_causal_skip=False)
                out2 = mtp.layers[0](R2, position_embeddings=pe2, attention_mask=mask2, past_key_values=None)
                pred2 = model.lm_head(mtp.hyper_connection_mixer(out2)).argmax(-1)[0]  # row t predicts x[t+3]
                t3 = seq[0, 3:]
                rows = range(gen0, T - 3)
                both = sum(int(pred[t] == seq[0, t + 2] and pred2[t] == t3[t]) for t in rows)
                d2[0] += both; d2[1] += len(rows)
        print(f"prompt done: shift0 {tot[0][0]}/{tot[0][1]}, shift1 {tot[1][0]}/{tot[1][1]}", flush=True)
print(f"MTP depth-2 (chained, both drafts right): {d2[0] / d2[1]:.3f}  -> expected tokens/step k=2: {1 + tot[0][0] / tot[0][1] + d2[0] / d2[1]:.2f}")
for s in (0, 1):
    print(f"MTP depth-1 acceptance (positions t+{s}): {tot[s][0] / tot[s][1]:.3f}")
