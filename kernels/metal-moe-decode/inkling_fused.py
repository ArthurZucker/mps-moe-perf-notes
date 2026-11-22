"""Fused Metal kernels for the small-op chains of Inkling-2 / Inkling-2-Loop at batch-1 decode.

Each op below replaces one chain of a decoder layer's forward that torch.compile lowers to several tiny kernels (and
whose cost on Metal is therefore the dispatch count, not the bytes) with a single dispatch. The kernels live in
`inkling_fused.metal`, compiled once with `torch.mps.compile_shader`; the ops are registered with the low-level
`torch.library.Library` API plus a fake, so they trace through torch.compile as opaque calls. (`custom_op` would add
~4 us of Python dispatch per call, as long as the kernels themselves.)

    attn_res(block_states, partial, query, eps=1e-6)                                  -> fp32 [..., n_queries, hidden]
    attn_res_rmsnorm(block_states, partial, query, weight, norm_eps, eps=1e-6)        -> [..., n_queries, hidden]
    message_gate(hidden_states, message, down_weight, forget_weight, message_weight, eps)         -> fp32 [..., hidden]
    message_gate_rmsnorm(..., eps, norm_weight, norm_eps)                                         -> [..., hidden]
    short_conv_update(x, conv_state, weight, residual, out_dtype)                                 -> [..., channels]
    attention_inputs(q, k, v, rel, q_scale, rel_scale, k_state, v_state, k_weight, v_weight,
                     k_dynamic, v_dynamic, position_ids, qk_threshold, rel_threshold, eps, log_alpha, log_n_floor)
                                                                                 -> [rows, q | k | v | rel] (flat)

Shapes outside what a kernel handles (prefill rows for the message gate, more than 8 stacked states, ...) fall back
to the equivalent torch code, so every op is correct for any input; the kernels cover decode.
"""

import functools
import math
import pathlib

import torch

from .bmv import bmv_multi
from .qmv4 import qmv4_multi


_NAMES = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader((pathlib.Path(__file__).parent / "inkling_fused.metal").read_text())
    return _LIB


@functools.cache
def _kernel(name):
    return getattr(_lib(), name)


def _rms_norm(x, weight, eps, dtype):
    """Inkling2RMSNorm on an fp32 input, then `.to(dtype)` as the decoder layers do."""
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * x).to(dtype)


# ---------------------------------------------------------------------------------------------------------------------
# Block / message attention residual
# ---------------------------------------------------------------------------------------------------------------------

_AR_MAX_STATES, _AR_PER_THREAD = 8, 3


def _attn_res_torch(block_states, partial, query, eps):
    hidden = partial.shape[-1]
    stacked = torch.cat([block_states, partial.unsqueeze(-2)], dim=-2).float()
    stacked = stacked * torch.rsqrt(stacked.square().mean(-1, keepdim=True) + eps)
    scores = torch.einsum("qh,...nh->...qn", query.float().reshape(-1, hidden), stacked) * hidden**-0.5
    return torch.einsum("...qn,...nh->...qh", scores.softmax(-1), stacked)


def _attn_res(block_states, partial, query, weight, eps, norm_eps):
    # kept lean: at decode this host path runs ~120 times per token
    hidden = partial.shape[-1]
    norm = weight is not None
    launch = _attn_res_launch(block_states.dtype, partial.dtype, norm, block_states.shape[-2] + 1, hidden)
    if launch is None or query.dtype != torch.float32 or partial.shape[:-1] != block_states.shape[:-2]:
        out = _attn_res_torch(block_states, partial, query, eps)
        return _rms_norm(out, weight.view(-1, hidden), norm_eps, block_states.dtype) if norm else out
    kernel, threads = launch
    n_queries, rows = query.numel() // hidden, partial.numel() // hidden
    out = torch.empty(
        *partial.shape[:-1],
        n_queries,
        hidden,
        dtype=block_states.dtype if norm else torch.float32,
        device=partial.device,
    )
    if rows == 0:
        return out
    if norm and weight.dtype != block_states.dtype:
        weight = weight.to(block_states.dtype)
    kernel(
        block_states.contiguous() if block_states.numel() else partial,  # with no block states it is never read
        partial.contiguous(),
        query.contiguous(),
        weight.contiguous() if norm else query,
        out,
        hidden,
        eps,
        norm_eps,
        hidden if norm and weight.numel() > hidden else 0,
        threads=[n_queries * threads, rows],
        group_size=[threads, 1],
    )
    return out


@functools.cache
def _attn_res_launch(block_dtype, partial_dtype, norm, n_states, hidden):
    """(kernel, threadgroup size) for one shape class, or None when the torch path must run. Resolved once: the decode
    loop only pays the dispatch."""
    if (
        block_dtype not in _NAMES
        or partial_dtype not in (block_dtype, torch.float32)
        or n_states > _AR_MAX_STATES
        or hidden > _AR_PER_THREAD * 1024
    ):
        return None
    name = f"attn_res_{_NAMES[block_dtype]}_{_NAMES[partial_dtype]}_{int(norm)}_{n_states}"
    return getattr(_lib(), name), min(1024, -(-hidden // (_AR_PER_THREAD * 32)) * 32)


def _attn_res_mps(block_states, partial, query, eps=1e-6):
    return _attn_res(block_states, partial, query, None, eps, 0.0)


def _attn_res_rmsnorm_mps(block_states, partial, query, weight, norm_eps, eps=1e-6):
    return _attn_res(block_states, partial, query, weight, eps, norm_eps)


# ---------------------------------------------------------------------------------------------------------------------
# Loop message gate
# ---------------------------------------------------------------------------------------------------------------------

# fixed grid of the message gate kernel (see inkling_fused.metal): threadgroups, threads, gate rank, column slices
_MG_TGS, _MG_THREADS, _MG_RANK, _MG_SPLIT = 16, 1024, 64, 8
_SCRATCH = {}


def _scratch(device):
    """(grid sync flags, cross-threadgroup exchange buffer) of the message gate kernel, shared by every call on
    `device`: calls are serialized on the stream, and the flags stay equal between calls."""
    if device not in _SCRATCH:
        _SCRATCH[device] = (
            torch.zeros(_MG_TGS, dtype=torch.int32, device=device),
            torch.zeros(_MG_RANK * _MG_SPLIT + _MG_TGS, dtype=torch.float32, device=device),
        )
    return _SCRATCH[device]


@functools.cache
def _message_gate_kernel(down_dtype, norm, hidden, rank):
    """The kernel for one shape class, or None when the torch path must run."""
    if down_dtype not in _NAMES or rank != _MG_RANK:
        return None
    return getattr(_lib(), f"message_gate_{_NAMES[down_dtype]}_{int(norm)}_{hidden}", None)


def _message_gate_torch(hidden_states, message, down_weight, forget_weight, message_weight, eps):
    hidden_states, message = hidden_states.float(), message.float()

    def norm(x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)

    gate_input = torch.cat([norm(hidden_states), norm(message)], dim=-1).to(down_weight.dtype)
    gate_input = torch.nn.functional.linear(gate_input, down_weight).float()
    forget_gate = torch.tanh(torch.nn.functional.linear(gate_input, forget_weight.float()))
    message_gate = torch.tanh(torch.nn.functional.linear(gate_input, message_weight.float()))
    return (1 - forget_gate) * hidden_states + message_gate * message


def _message_gate(hidden_states, message, down_weight, forget_weight, message_weight, eps, norm_weight, norm_eps):
    hidden = hidden_states.shape[-1]
    norm = norm_weight is not None
    model_dtype = down_weight.dtype
    kernel = _message_gate_kernel(model_dtype, norm, hidden, down_weight.shape[0])
    if kernel is None or hidden_states.numel() != hidden or message.numel() != hidden:
        out = _message_gate_torch(hidden_states, message, down_weight, forget_weight, message_weight, eps)
        return _rms_norm(out, norm_weight, norm_eps, model_dtype) if norm else out
    flags, exchange = _scratch(hidden_states.device)
    out = torch.empty(hidden_states.shape, dtype=model_dtype if norm else torch.float32, device=hidden_states.device)
    kernel(
        hidden_states.float().contiguous(),
        message.float().contiguous(),
        down_weight.contiguous(),
        forget_weight.float().contiguous(),
        message_weight.float().contiguous(),
        norm_weight.to(model_dtype).contiguous() if norm else down_weight,
        out,
        flags,
        exchange,
        eps,
        norm_eps,
        threads=_MG_TGS * _MG_THREADS,
        group_size=_MG_THREADS,
    )
    return out


def _message_gate_mps(hidden_states, message, down_weight, forget_weight, message_weight, eps):
    return _message_gate(hidden_states, message, down_weight, forget_weight, message_weight, eps, None, 0.0)


def _message_gate_rmsnorm_mps(
    hidden_states, message, down_weight, forget_weight, message_weight, eps, norm_weight, norm_eps
):
    return _message_gate(
        hidden_states, message, down_weight, forget_weight, message_weight, eps, norm_weight, norm_eps
    )


# ---------------------------------------------------------------------------------------------------------------------
# Short convolution decode step + residual
# ---------------------------------------------------------------------------------------------------------------------


@functools.cache
def _short_conv_kernel(tx, tr, to, has_residual):
    """The instantiation for this dtype combination, or None (then the torch path runs)."""
    name = f"short_conv_update_{_NAMES.get(tx)}_{_NAMES.get(tr)}_{_NAMES.get(to)}_{int(has_residual)}"
    return getattr(_lib(), name, None)


def _short_conv_update_mps(x, conv_state, weight, residual, out_dtype):
    channels, kernel_size = conv_state.shape[-2], conv_state.shape[-1]
    rows = conv_state.numel() // (channels * kernel_size)
    if x.numel() != rows * channels:
        raise ValueError("short_conv_update is a single-token decode step: x must hold one token per state row")
    weight = weight.reshape(channels, kernel_size)
    tx = x.dtype
    kernel = _short_conv_kernel(tx, residual.dtype if residual is not None else tx, out_dtype, residual is not None)
    if kernel is None or conv_state.dtype != torch.float32 or not conv_state.is_contiguous():
        # same math in torch (any dtype combination / non-contiguous state)
        xf = x.float().reshape(rows, channels)
        state = torch.cat([conv_state.reshape(rows, channels, kernel_size)[..., 1:], xf.unsqueeze(-1)], dim=-1)
        conv_state.copy_(state.view_as(conv_state))
        y = ((state * weight.float()).sum(-1) + xf) / math.sqrt(kernel_size + 1)
        y = y.to(tx).float()
        if residual is not None:
            y = y + residual.float().reshape(rows, channels)
        return y.to(out_dtype).view(x.shape)
    out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    kernel(
        x.contiguous(),
        conv_state,
        weight.float().contiguous(),
        residual.contiguous() if residual is not None else x,
        out,
        channels,
        kernel_size,
        math.sqrt(kernel_size + 1),
        threads=[channels, rows],
        group_size=[min(256, channels), 1],
    )
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Attention input tails (q / k / v / relative states)
# ---------------------------------------------------------------------------------------------------------------------


def _attention_inputs_mps(
    q,
    k,
    v,
    rel,
    q_scale,
    rel_scale,
    k_state,
    v_state,
    k_weight,
    v_weight,
    k_dynamic,
    v_dynamic,
    position_ids,
    qk_threshold,
    rel_threshold,
    eps,
    log_alpha,
    log_n_floor,
):
    q_heads, head_dim = q_scale.shape[-2], q_scale.shape[-1]
    rel_dim = rel_scale.shape[-1]
    channels, kernel_size = k_state.shape[-2], k_state.shape[-1]
    kv_heads = channels // head_dim
    rows = k_state.numel() // (channels * kernel_size)
    dynamic = k_dynamic is not None
    dtype = q.dtype
    if (
        q.numel() != rows * q_heads * head_dim
        or head_dim > 128
        or rel_dim > 128
        or kernel_size > 8
        or k_state.dtype != torch.float32
        or v_state.dtype != torch.float32
    ):
        raise ValueError("attention_inputs is a single-token decode step with head_dim <= 128 and fp32 conv states")
    if dynamic and position_ids is None:
        raise ValueError("the full-attention (dynamic convolution) variant needs position_ids for the log scale")
    tw = k_weight.dtype if dynamic and k_weight.dtype in (dtype, torch.float32) else torch.float32
    out = torch.empty(rows, (q_heads + 2 * kv_heads) * head_dim + q_heads * rel_dim, dtype=dtype, device=q.device)
    _kernel(f"attention_inputs_{_NAMES[dtype]}_{_NAMES[tw]}_{int(dynamic)}")(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        rel.contiguous(),
        q_scale.to(dtype).contiguous(),
        rel_scale.to(dtype).contiguous(),
        k_state,
        v_state,
        k_weight.to(tw).contiguous(),
        v_weight.to(tw).contiguous(),
        k_dynamic.contiguous() if dynamic else q,
        v_dynamic.contiguous() if dynamic else q,
        position_ids.reshape(-1).contiguous() if dynamic else k_state,
        out,
        q_heads,
        kv_heads,
        head_dim,
        rel_dim,
        kernel_size,
        qk_threshold,
        rel_threshold,
        eps,
        log_alpha,
        log_n_floor,
        threads=[32 * (2 * q_heads + 2 * kv_heads), rows],
        group_size=[32, 1],
    )
    return out


def split_attention_inputs(out, q_heads, kv_heads, head_dim, rel_dim):
    """Views of `attention_inputs`' flat output in the layouts the attention forward expects: queries / keys / values
    [batch, heads, 1, head_dim] (as returned by the q / k / v projections) and relative states [batch, 1, q_heads,
    rel_dim] (as fed to `relative_states @ proj`)."""
    batch = out.shape[0]
    q, k, v, rel = out.split([q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim, q_heads * rel_dim], -1)
    return (
        q.view(batch, 1, q_heads, head_dim).transpose(1, 2),
        k.view(batch, 1, kv_heads, head_dim).transpose(1, 2),
        v.view(batch, 1, kv_heads, head_dim).transpose(1, 2),
        rel.view(batch, 1, q_heads, rel_dim),
    )


# ---------------------------------------------------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------------------------------------------------

_OPS = torch.library.Library("inkling_fused", "DEF")
_OPS.define("attn_res(Tensor block_states, Tensor partial, Tensor query, float eps=1e-06) -> Tensor")
_OPS.define(
    "attn_res_rmsnorm(Tensor block_states, Tensor partial, Tensor query, Tensor weight, float norm_eps, "
    "float eps=1e-06) -> Tensor"
)
_OPS.define(
    "message_gate(Tensor hidden_states, Tensor message, Tensor down_weight, Tensor forget_weight, "
    "Tensor message_weight, float eps) -> Tensor"
)
_OPS.define(
    "message_gate_rmsnorm(Tensor hidden_states, Tensor message, Tensor down_weight, Tensor forget_weight, "
    "Tensor message_weight, float eps, Tensor norm_weight, float norm_eps) -> Tensor"
)
_OPS.define(
    "short_conv_update(Tensor x, Tensor(a!) conv_state, Tensor weight, Tensor? residual, ScalarType out_dtype) "
    "-> Tensor"
)
_OPS.define(
    "attention_inputs(Tensor q, Tensor k, Tensor v, Tensor rel, Tensor q_scale, Tensor rel_scale, "
    "Tensor(a!) k_state, Tensor(b!) v_state, Tensor k_weight, Tensor v_weight, Tensor? k_dynamic, "
    "Tensor? v_dynamic, Tensor? position_ids, float qk_threshold, float rel_threshold, float eps, float log_alpha, "
    "float log_n_floor) -> Tensor"
)
_OPS.impl("attn_res", _attn_res_mps, "MPS")
_OPS.impl("attn_res_rmsnorm", _attn_res_rmsnorm_mps, "MPS")
_OPS.impl("message_gate", _message_gate_mps, "MPS")
_OPS.impl("message_gate_rmsnorm", _message_gate_rmsnorm_mps, "MPS")
_OPS.impl("short_conv_update", _short_conv_update_mps, "MPS")
_OPS.impl("attention_inputs", _attention_inputs_mps, "MPS")


@torch.library.register_fake("inkling_fused::attn_res")
def _(block_states, partial, query, eps=1e-6):
    hidden = partial.shape[-1]
    return partial.new_empty(*partial.shape[:-1], query.numel() // hidden, hidden, dtype=torch.float32)


@torch.library.register_fake("inkling_fused::attn_res_rmsnorm")
def _(block_states, partial, query, weight, norm_eps, eps=1e-6):
    hidden = partial.shape[-1]
    return partial.new_empty(*partial.shape[:-1], query.numel() // hidden, hidden, dtype=block_states.dtype)


@torch.library.register_fake("inkling_fused::message_gate")
def _(hidden_states, message, down_weight, forget_weight, message_weight, eps):
    return hidden_states.new_empty(hidden_states.shape, dtype=torch.float32)


@torch.library.register_fake("inkling_fused::message_gate_rmsnorm")
def _(hidden_states, message, down_weight, forget_weight, message_weight, eps, norm_weight, norm_eps):
    return hidden_states.new_empty(hidden_states.shape, dtype=down_weight.dtype)


@torch.library.register_fake("inkling_fused::short_conv_update")
def _(x, conv_state, weight, residual, out_dtype):
    return x.new_empty(x.shape, dtype=out_dtype)


@torch.library.register_fake("inkling_fused::attention_inputs")
def _(q, k, v, rel, q_scale, rel_scale, k_state, v_state, *args):
    q_heads, head_dim = q_scale.shape[-2], q_scale.shape[-1]
    channels, kernel_size = k_state.shape[-2], k_state.shape[-1]
    rows = k_state.numel() // (channels * kernel_size)
    width = (q_heads + 2 * (channels // head_dim)) * head_dim + q_heads * rel_scale.shape[-1]
    return q.new_empty(rows, width)


attn_res = torch.ops.inkling_fused.attn_res.default
"""attn_res(block_states [..., n, H], partial [..., H], query fp32 [n_queries, H] or [H], eps=1e-6) -> fp32
[..., n_queries, H]. `Inkling2BlockAttentionResidual(block_states, partial)` is `attn_res(...).unbind(-2)`;
`Inkling2LoopMessageAttentionResidual(block_states, partial, i)` is `attn_res(block_states, partial, query[i])`."""

attn_res_rmsnorm = torch.ops.inkling_fused.attn_res_rmsnorm.default
"""attn_res_rmsnorm(block_states, partial, query, weight [H] or [n_queries, H], norm_eps, eps=1e-6) -> [..., n_queries,
H] in block_states' dtype: `Inkling2RMSNorm(weight_i)(attn_res(...)[..., i, :]).to(block_states.dtype)` for each query
(one weight shared by all queries, or one row per query: q / k / v input layernorms stacked)."""

message_gate = torch.ops.inkling_fused.message_gate.default
"""message_gate(hidden_states [..., H], message [..., H], down_weight [64, 2H], forget_weight fp32 [H, 64],
message_weight fp32 [H, 64], eps) -> fp32 [..., H]: `Inkling2LoopMessageGate.forward`."""

message_gate_rmsnorm = torch.ops.inkling_fused.message_gate_rmsnorm.default
"""message_gate_rmsnorm(..., eps, norm_weight [H], norm_eps) -> [..., H] in down_weight's dtype:
`Inkling2RMSNorm(norm_weight)(message_gate(...)).to(down_weight.dtype)`."""

short_conv_update = torch.ops.inkling_fused.short_conv_update.default
"""short_conv_update(x [b, 1, C], conv_state fp32 [b, C, K] (updated in place), weight fp32 [C, K] or [C, 1, K],
residual [b, 1, C] or None, out_dtype) -> out_dtype [b, 1, C]: `Inkling2ShortConvolution` decode step (rounded to
x.dtype, as the module returns it), plus `residual` in fp32, cast to out_dtype."""

attention_inputs = torch.ops.inkling_fused.attention_inputs.default
"""attention_inputs(q, k, v, rel, q_scale [Hq, D], rel_scale [Hq, d_rel], k_state, v_state, k_weight, v_weight,
k_dynamic, v_dynamic, position_ids, qk_threshold, rel_threshold, eps, log_alpha, log_n_floor) -> flat [b, Hq*D + 2*Hkv*D
+ Hq*d_rel]; see `split_attention_inputs`. q / k / v / rel are the raw projection outputs of one token. Sliding layers:
k_dynamic = v_dynamic = position_ids = None, k_weight / v_weight the fp32 sconv.conv1d weights. Full layers: k_dynamic /
v_dynamic the sconv.weight_proj outputs [b, 1, Hkv*K], k_weight / v_weight the sconv.static_weight [K, C], and the log
scale of the queries / relative states is applied from position_ids."""


def inkling2_attention_residual(block_states, partial_block_states, query, module):
    """transformers' `inkling2_attention_residual` on MPS: RMSNorm of the stacked block states, attention with the
    learned queries and the weighted sum in one dispatch -> [..., n_queries, hidden] float32."""
    return attn_res(block_states, partial_block_states, query, module.norm.eps)


def inkling2_message_gate(hidden_states, message, module):
    """transformers' `inkling2_message_gate` on MPS: `(1 - forget) * hidden_states + gain * message`, gates from the
    normalized pair through the rank-64 projections, in one dispatch -> float32."""
    return message_gate(
        hidden_states, message, module.down_proj.weight, module.forget_gate_proj.weight,
        module.message_gate_proj.weight, module.norm.eps,
    )  # fmt: skip


def inkling2_short_convolution_step(hidden_states, conv_state, weight, conv_kernel_size):
    """transformers' `inkling2_short_convolution_step` on MPS: conv state shift, depthwise taps, `(conv + x) / sqrt(k + 1)`
    in one dispatch."""
    return short_conv_update(hidden_states, conv_state, weight, None, hidden_states.dtype)



def _quantized(linear):
    """(weight, scales, biases, group size) of a 4-bit MetalLinear-like module, else None."""
    w = getattr(linear, "weight", None)
    if w is None or w.dtype != torch.uint32 or getattr(linear, "bits", None) != 4 or getattr(linear, "bias", None) is not None:
        return None
    return w, linear.scales, linear.qbiases, linear.group_size


def _project_qkv(module, hidden_states, key_hidden_states, value_hidden_states):
    """Raw q / k / v projections of one decode token. When q is the first member of a fused sibling group (its
    `fused` 4-bit linear computes q and the other projections of `hidden_states`) and k / v are 4-bit linears of the
    same input size, the three matvecs run as one `qmv4_multi` dispatch and the group's cache is filled so its other
    members slice the fused output."""
    q, k, v = module.q_proj.linear, module.k_proj.linear, module.v_proj.linear
    fused = getattr(q, "fused", None)
    parts = [_quantized(m) for m in (fused, k, v)] if fused is not None and getattr(q, "start", 1) == 0 else None
    if (
        parts is not None
        and all(p is not None for p in parts)
        and len({(p[0].shape[1], p[3]) for p in parts}) == 1
        and hidden_states.dtype in (torch.bfloat16, torch.float16)
        and all(p[1].dtype == hidden_states.dtype for p in parts)
    ):
        g = parts[0][3]
        out, key, value = qmv4_multi(
            hidden_states, *parts[0][:3], key_hidden_states.to(hidden_states.dtype), *parts[1][:3],
            value_hidden_states.to(hidden_states.dtype), *parts[2][:3], g,
        )  # fmt: skip
        q.state["input"], q.state["output"] = hidden_states, out
        return q(hidden_states), key, value
    plain = [m for m in (q, k, v) if type(m) is torch.nn.Linear]
    if (
        len(plain) == 3
        and all(m.bias is None and m.weight.dtype == hidden_states.dtype for m in plain)
        and hidden_states.dtype in (torch.bfloat16, torch.float16)
        and len({m.in_features for m in plain}) == 1
        and plain[0].in_features % 8 == 0
    ):  # exact (unquantized) weights: the three bf16 matvecs in one dispatch
        return bmv_multi(
            hidden_states, q.weight, key_hidden_states.to(hidden_states.dtype), k.weight,
            value_hidden_states.to(hidden_states.dtype), v.weight,
        )  # fmt: skip
    return q(hidden_states), k(key_hidden_states), v(value_hidden_states)


def inkling2_attention_inputs(
    hidden_states, key_hidden_states, value_hidden_states, module, cache, cache_offset, conv_mask, **kwargs
):
    """transformers' `inkling2_attention_inputs` on MPS: for one decode token, the projection tails (query / relative
    state L2 clips and log scale, key / value short convolutions + clip / RMSNorm, conv states advanced in place) in
    one dispatch after the raw projections; anything else takes the model's own path."""
    position_ids = kwargs["position_ids"]
    k_conv, v_conv = module.k_proj.sconv, module.v_proj.sconv
    layer_idx = module.layer_idx + cache_offset
    single = hidden_states.shape[0] == 1 and hidden_states.shape[1] == 1 and conv_mask is None and cache is not None
    if single and not cache.is_recording_past(layer_idx):
        k_state = cache.get_conv_state(layer_idx, state_idx=k_conv.conv_idx)
        v_state = cache.get_conv_state(layer_idx, state_idx=v_conv.conv_idx)
    else:
        k_state = v_state = None
    if k_state is None or v_state is None:
        query_states = module.q_proj(hidden_states, position_ids)
        key_states = module.k_proj(key_hidden_states, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs)
        value_states = module.v_proj(
            value_hidden_states, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs
        )
        return query_states, key_states, value_states, module.relative_position_bias.relative_states(hidden_states, position_ids)

    config, bias = module.config, module.relative_position_bias
    if hasattr(k_conv, "weight_proj"):  # full layers: dynamic short convolutions, log-scaled queries / relative states
        dynamic = (k_conv.weight_proj(key_hidden_states), v_conv.weight_proj(value_hidden_states), position_ids)
        weights = (k_conv.static_weight, v_conv.static_weight)
    else:
        dynamic, weights = (None, None, None), (k_conv.conv1d.weight, v_conv.conv1d.weight)
    flat = attention_inputs(
        *_project_qkv(module, hidden_states, key_hidden_states, value_hidden_states), bias.linear(hidden_states), module.q_proj.norm.scale,
        bias.norm.scale, k_state, v_state, *weights, *dynamic, config.qk_l2_clip_threshold,
        config.rel_l2_clip_threshold, config.rms_norm_eps, config.log_scaling_alpha, float(config.log_scaling_n_floor),
    )  # fmt: skip
    return split_attention_inputs(flat, module.num_heads, module.num_key_value_heads, module.head_dim, config.d_rel)
