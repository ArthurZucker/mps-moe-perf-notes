import os, torch
from kernels import get_local_kernel
from pathlib import Path
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual
from transformers import AutoConfig
import glob
k = get_local_kernel(Path(os.environ["TRANSFORMERS_MOE_AFFINE_KERNEL"]))
cfg = AutoConfig.from_pretrained(glob.glob('/raid/arthur/hub/models--ArthurZ--Qwen3.8-Flash-Next-metal-experts-4bit/snapshots/*')[0]).text_config
torch.manual_seed(0)
m = Qwen4ExpTextGatedResidual(cfg).cuda().bfloat16()
for p in m.parameters(): p.data.normal_(0, 0.02)
H, G = cfg.hidden_size, cfg.hc_count
x = torch.randn(1, G * H, device="cuda").bfloat16() * 3
ref = m(x)[0].float()
xhat = torch.empty(H, device="cuda").bfloat16(); scratch = torch.empty(2 * G * H + cfg.hc_lowrank, device="cuda")
k.hc_mix_guess(x.contiguous(), m.hc_norm.weight, m.input_mix_weight_down.weight, m.input_mix_weight_up.weight, cfg.rms_norm_eps, xhat, scratch)
print("hc_mix_guess rel err vs module:", ((xhat.float() - ref.reshape(-1)).norm() / ref.norm()).item())
