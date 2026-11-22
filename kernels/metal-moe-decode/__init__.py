"""Metal kernels for MoE / hybrid-attention models on Apple Silicon (shaders compiled at import with
`torch.mps.compile_shader`; every op is registered with a fake, so `torch.compile` keeps it in graph).

- `qmv4`: batch-1 4-bit affine matvec (MLX packing, groups of 32 / 64), a layout per shape
- `experts_decode`: one token through its routed 4-bit affine experts (gate/up + SwiGLU, then down + routing sum);
  `experts_down_sum`, `experts_gate_up_masked`, `experts_gate_up_swiglu` are its parts
- `fp4_gate_up_act`, `fp4_down_sum`: routed experts straight from NVFP4 / MXFP8 checkpoint banks ([gate; up] rows,
  softcapped or plain SwiGLU in-kernel), any number of tokens, never dequantized
- `gated_delta_rule`, `causal_conv1d_fn`, `causal_conv1d_update`: linear-attention ops (same contracts as FLA /
  causal-conv1d), states advanced in place; `gated_delta_net_decode` fuses a GatedDeltaNet decode step
- `decode_attention`: one-token attention with Inkling2's relative position bias + sinks + GQA in one dispatch;
  `inkling2_attention` adapts it to transformers' Inkling2 attention
- `inkling2_attention_residual`: Inkling2's block attention residual (stacked-state RMSNorm + learned-query attention)
  in one dispatch; `inkling2_message_gate`: Inkling2 Mini's loop message gate in two
"""

from .experts import experts_decode, experts_down_sum, experts_gate_up_masked, experts_gate_up_swiglu
from .fp4_experts_opt import down_route as fp4_down_route
from .fp4_experts_opt import down_sum as fp4_down_sum
from .fp4_experts_opt import experts_decode as fp4_experts_decode
from .fp4_experts_opt import gate_up_act as fp4_gate_up_act
from .fp4_gemm import experts_prefill as fp4_experts_prefill
from .inkling_attention import decode_attention, inkling2_attention
from .inkling_fused import (
    inkling2_attention_inputs,
    inkling2_attention_residual,
    inkling2_message_gate,
    inkling2_short_convolution_step,
)
from .linear_attention import causal_conv1d_fn, causal_conv1d_update, gated_delta_net_decode, gated_delta_rule
from .bmv import bmv_multi
from .qmv4 import qmv4
from .router import inkling2_router, router_topk


__all__ = [
    "qmv4",
    "experts_decode",
    "experts_down_sum",
    "experts_gate_up_masked",
    "experts_gate_up_swiglu",
    "fp4_gate_up_act",
    "fp4_down_sum",
    "fp4_down_route",
    "fp4_experts_decode",
    "fp4_experts_prefill",
    "gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "gated_delta_net_decode",
    "decode_attention",
    "inkling2_attention",
    "inkling2_attention_residual",
    "inkling2_message_gate",
    "inkling2_short_convolution_step",
    "inkling2_attention_inputs",
    "router_topk",
    "inkling2_router",
]
