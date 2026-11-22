"""Metal (Apple Silicon) kernels for the linear-attention layers of hybrid models: the gated delta rule (Qwen3-Next
style gated delta nets) and the short depthwise causal conv that feeds it.

Public wrappers keep the flash-linear-attention / causal-conv1d contracts:

    causal_conv1d_fn(x, weight, bias, activation)                  prefill conv, x [B, C, L] (any strides)
    causal_conv1d_update(x, conv_state, weight, bias, activation)  one decode token, conv_state shifted in place
    gated_delta_rule(q, k, v, g, beta, initial_state, ...)         decode or prefill, state advanced in place

plus one fused decode entry point, `gated_delta_net_decode`, that runs a whole gated delta net core for one token in
a single dispatch (conv step, q/k/v split, q/k l2 norm, decay / beta gates, delta rule).

Each kernel is also registered as a `torch.library.custom_op` with a fake impl so `torch.compile` keeps it in the
graph; in eager mode the wrappers call the launchers directly (the custom-op dispatcher costs more host time than the
kernels take on the GPU).
"""

import pathlib

import torch


_SOURCE = (pathlib.Path(__file__).parent / "linear_attention.metal").read_text()
_NAMES = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}
_LIB = None
_KERNELS: dict = {}

# Row slices per state column (RS): a simdgroup covers 4 * 32 / RS columns, each thread DK / RS rows of 4 columns.
# Decode is one pass over the state (bandwidth-bound): RS only sets the per-thread tile and barely matters (8 / 16 /
# 32 within 10%). Prefill is throughput-bound: RS 16 balances the shuffle count against registers (RS 8 spills, RS 32
# shuffles too much). The fused decode kernel also convolves the thread's q/k rows, which favours fewer, longer
# slices. Tuned on an M3 Max.
DECODE_ROW_SLICES = {64: 16, 128: 16, 256: 16}
FUSED_ROW_SLICES = {64: 8, 128: 8, 256: 16}
PREFILL_ROW_SLICES = {64: 16, 128: 16, 256: 32}
SIMDGROUPS_PER_THREADGROUP = 4
# From this many tokens on, the chunk-staged prefill kernel (q/k/v staged through threadgroup memory) is used.
CHUNKED_MIN_TOKENS = 16


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_SOURCE)
    return _LIB


def _kernel(name: str):
    """Kernel handle by host name, cached (the lookup is a measurable share of the ~10 us host cost of a launch)."""
    fn = _KERNELS.get(name)
    if fn is None:
        fn = _KERNELS[name] = getattr(_lib(), name)
    return fn


def _silu(activation) -> int:
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"Metal causal conv1d supports silu or no activation, got {activation}")
    return int(activation is not None)


def _eager() -> bool:
    return not torch.compiler.is_compiling()


def _delta_rule_grid(batch: int, v_heads: int, k_dim: int, v_dim: int, row_slices: int):
    if k_dim % (4 * row_slices) or v_dim % (128 // row_slices):
        raise ValueError(f"Metal gated delta rule: unsupported head dims (k {k_dim}, v {v_dim})")
    simdgroups = v_heads * v_dim // (128 // row_slices)
    groups = -(-simdgroups // SIMDGROUPS_PER_THREADGROUP)
    width = 32 * SIMDGROUPS_PER_THREADGROUP
    return {"threads": (groups * width, batch), "group_size": (width, 1)}


def _vec_view(x: torch.Tensor) -> torch.Tensor:
    """`x` [B, L, H, D] itself when its heads and dims are contiguous and its rows 8-byte aligned (the kernels load 4
    elements at a time and step through batch and time by strides), else a contiguous copy."""
    _, _, heads, dim = x.shape
    ok = x.stride(-1) == 1 and (heads == 1 or x.stride(-2) == dim)
    ok = ok and x.stride(0) % 4 == 0 and x.stride(1) % 4 == 0 and x.storage_offset() % 4 == 0
    return x if ok else x.contiguous()


# ------------------------------------------------------------------------------------------------- launchers


def _conv_impl(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, silu: int) -> torch.Tensor:
    batch, channels, length = x.shape
    out = torch.empty_like(x)  # keeps x's layout (a channels-last view stays channels-last)
    c_fast = int(x.stride(1) == 1)
    threads = (channels, length, batch) if c_fast else (length, channels, batch)
    _kernel(f"causal_conv1d_fwd_{_NAMES[x.dtype]}")(
        x, weight, weight if bias is None else bias, out, channels, length, weight.shape[-1], int(bias is not None),
        silu, *x.stride(), *out.stride(), c_fast, threads=threads, group_size=(min(threads[0], 256), 1, 1),
    )  # fmt: skip
    return out


def _conv_step_impl(x, state, weight, bias, silu):
    batch, channels, _ = x.shape
    out = torch.empty_like(x)
    _kernel(f"causal_conv1d_step_{_NAMES[x.dtype]}_{_NAMES[state.dtype]}")(
        x, state, weight, weight if bias is None else bias, out, channels, state.shape[-1], weight.shape[-1],
        int(bias is not None), silu, threads=(channels, batch), group_size=(min(channels, 256), 1),
    )  # fmt: skip
    return out


def _gdr_impl(q, k, v, g, beta, state, l2norm):
    batch, length, k_heads, k_dim = k.shape
    v_heads, v_dim = v.shape[-2:]
    out = torch.empty(batch, length, v_heads, v_dim, device=v.device, dtype=v.dtype)
    q, k, v = _vec_view(q), _vec_view(k), _vec_view(v)
    chunked = length >= CHUNKED_MIN_TOKENS
    rs = (PREFILL_ROW_SLICES if chunked else DECODE_ROW_SLICES)[k_dim]
    if chunked:  # 128 threads per threadgroup, each threadgroup W columns of one head
        width = 4 * 4 * (32 // rs)
        if v_dim % width:
            raise ValueError(f"Metal gated delta rule: unsupported value head dim {v_dim}")
        grid = {"threads": (v_heads * v_dim // width * 128, batch), "group_size": (128, 1)}
    else:
        grid = _delta_rule_grid(batch, v_heads, k_dim, v_dim, rs)
    name = f"gated_delta_rule{'_chunked' if chunked else ''}_{_NAMES[v.dtype]}_{_NAMES[state.dtype]}_{k_dim}_{rs}"
    _kernel(name)(
        q, k, v, g, beta, state, out, length, k_heads, v_heads, v_dim, *q.stride()[:2], *k.stride()[:2],
        *v.stride()[:2], l2norm, **grid,
    )  # fmt: skip
    return out


_ARRIVALS: dict = {}


def _arrivals(device: torch.device, n: int) -> torch.Tensor:
    """Zeroed int32 arrival counters for the fused decode kernel; every launch leaves them zeroed again."""
    buf = _ARRIVALS.get(device)
    if buf is None or buf.numel() < n:
        buf = _ARRIVALS[device] = torch.zeros(max(n, 256), device=device, dtype=torch.int32)
    return buf


def _gdn_decode_impl(x, conv_state, conv_weight, conv_bias, a, b, A_log, dt_bias, state, k_heads, k_dim, v_heads,
                     v_dim, silu, l2norm):  # fmt: skip
    batch = x.shape[0]
    out = torch.empty(batch, 1, v_heads, v_dim, device=x.device, dtype=x.dtype)
    rs = FUSED_ROW_SLICES[k_dim]
    _kernel(f"gated_delta_net_decode_{_NAMES[x.dtype]}_{_NAMES[state.dtype]}_{k_dim}_{rs}")(
        x, conv_state, conv_weight, conv_weight if conv_bias is None else conv_bias, a, b, A_log, dt_bias, state, out,
        _arrivals(x.device, batch * k_heads), conv_state.shape[-1], conv_weight.shape[-1], int(conv_bias is not None),
        silu, k_heads, v_heads, v_dim, x.stride(0), l2norm, **_delta_rule_grid(batch, v_heads, k_dim, v_dim, rs),
    )  # fmt: skip
    return out


# --------------------------------------------------------------------------------------- custom ops (compile)


@torch.library.custom_op("metal_linear_attention::causal_conv1d", mutates_args=())
def _conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, silu: int) -> torch.Tensor:
    return _conv_impl(x, weight, bias, silu)


@_conv.register_fake
def _(x, weight, bias, silu):
    return torch.empty_like(x)


@torch.library.custom_op("metal_linear_attention::causal_conv1d_step", mutates_args=("state",))
def _conv_step(
    x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, silu: int
) -> torch.Tensor:
    return _conv_step_impl(x, state, weight, bias, silu)


@_conv_step.register_fake
def _(x, state, weight, bias, silu):
    return torch.empty_like(x)


@torch.library.custom_op("metal_linear_attention::gated_delta_rule", mutates_args=("state",))
def _gdr(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, state: torch.Tensor,
    l2norm: int,
) -> torch.Tensor:  # fmt: skip
    return _gdr_impl(q, k, v, g, beta, state, l2norm)


@_gdr.register_fake
def _(q, k, v, g, beta, state, l2norm):
    return v.new_empty(v.shape)


@torch.library.custom_op("metal_linear_attention::gated_delta_net_decode", mutates_args=("conv_state", "state"))
def _gdn_decode(
    x: torch.Tensor, conv_state: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor | None,
    a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor, state: torch.Tensor, k_heads: int,
    k_dim: int, v_heads: int, v_dim: int, silu: int, l2norm: int,
) -> torch.Tensor:  # fmt: skip
    return _gdn_decode_impl(x, conv_state, conv_weight, conv_bias, a, b, A_log, dt_bias, state, k_heads, k_dim,
                            v_heads, v_dim, silu, l2norm)  # fmt: skip


@_gdn_decode.register_fake
def _(x, conv_state, conv_weight, conv_bias, a, b, A_log, dt_bias, state, k_heads, k_dim, v_heads, v_dim, silu,
      l2norm):  # fmt: skip
    return x.new_empty(x.shape[0], 1, v_heads, v_dim)


# ------------------------------------------------------------------------------------------- public wrappers


def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, seq_idx=None, **kwargs):
    """Depthwise causal conv over [B, C, L] (`weight` [C, K]), zero left padding, optional SiLU. Any input strides with
    a unit stride on time or channels: the transposed in_proj output ([B, L, C] viewed as [B, C, L]) is consumed and
    returned in that layout, without copies."""
    if seq_idx is not None:
        raise ValueError("Metal causal conv1d does not keep packed sequences apart (`seq_idx`)")
    x = hidden_states.to(weight.dtype)
    if 1 not in x.stride()[1:]:
        x = x.contiguous()
    weight = weight.contiguous()
    bias = None if bias is None else bias.to(weight.dtype).contiguous()
    out = (_conv_impl if _eager() else _conv)(x, weight, bias, _silu(activation))
    return out.to(hidden_states.dtype)


def causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None, **kwargs):
    """One decode token: `hidden_states` [B, C, 1], `conv_state` [B, C, N] (N >= K) shifted left in place."""
    if hidden_states.shape[-1] != 1:
        raise ValueError("Metal causal_conv1d_update takes one token at a time")
    x = hidden_states.to(weight.dtype).contiguous()
    weight = weight.contiguous()
    bias = None if bias is None else bias.to(weight.dtype).contiguous()
    out = (_conv_step_impl if _eager() else _conv_step)(x, conv_state, weight, bias, _silu(activation))
    return out.to(hidden_states.dtype)


def _state_for(initial_state, batch, v_heads, k_dim, v_dim, value):
    if initial_state is None:
        # In the activations' dtype: the cache keeps it (half the traffic of float32); the kernel computes in float32
        # and rounds once per token (no drift over thousands of tokens, see test.py)
        return torch.zeros(batch, v_heads, k_dim, v_dim, device=value.device, dtype=value.dtype)
    if initial_state.dtype in (torch.float32, value.dtype) and initial_state.is_contiguous():
        return initial_state  # advanced in place: the cache's own buffer
    return initial_state.float().contiguous()


def gated_delta_rule(
    query, key, value, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False,
    cu_seqlens=None, **kwargs,
):  # fmt: skip
    """Same contract as FLA's `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` (and transformers'
    `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule`): q, k [B, L, HK, DK], v [B, L, HV, DV],
    g (log decay) and beta [B, L, HV], state [B, HV, DK, DV]. Returns (out [B, L, HV, DV], final state or None).

    One kernel for decode and prefill: the state stays in registers across the sequence. q/k/v may be strided views
    along batch and time (e.g. the split of the conv output); they are not copied."""
    if cu_seqlens is not None:
        raise ValueError("Metal gated delta rule does not keep packed sequences apart (`cu_seqlens`)")
    batch, _, _, k_dim = key.shape
    v_heads, v_dim = value.shape[-2:]
    dtype = value.dtype
    state = _state_for(initial_state, batch, v_heads, k_dim, v_dim, value)
    query, key = query.to(dtype), key.to(dtype)  # strided views are fine (see `_vec_view`)
    beta, g = beta.to(dtype).contiguous(), g.float().contiguous()
    out = (_gdr_impl if _eager() else _gdr)(query, key, value, g, beta, state, int(use_qk_l2norm_in_kernel))
    return out, state if output_final_state else None


def gated_delta_net_decode(
    mixed_qkv, conv_state, conv_weight, conv_bias, a, b, A_log, dt_bias, recurrent_state, num_k_heads, num_v_heads,
    head_k_dim, head_v_dim, activation="silu", use_qk_l2norm=True,
):  # fmt: skip
    """One decode token of a gated delta net core (Qwen3-Next / Qwen3.5 `GatedDeltaNet.forward` between the input
    projections and the output norm), in one dispatch. Equivalent to:

        qkv = causal_conv1d_update(mixed_qkv.transpose(1, 2), conv_state, conv_weight, conv_bias, activation)
        q, k, v = qkv.transpose(1, 2).split([HK * DK, HK * DK, HV * DV], -1)       # reshaped to heads
        g = -A_log.float().exp() * softplus(a.float() + dt_bias);  beta = b.sigmoid()
        out, _ = gated_delta_rule(q, k, v, g, beta, recurrent_state, use_qk_l2norm_in_kernel=use_qk_l2norm)

    Args:
        mixed_qkv: [B, 1, 2 * HK * DK + HV * DV], the in_proj_qkv output (channel order q | k | v).
        conv_state: [B, conv_dim, N] (N >= K), same dtype as `mixed_qkv`; shifted in place.
        conv_weight: [conv_dim, K] (the depthwise Conv1d weight squeezed); conv_bias: [conv_dim] or None.
        a, b: [B, 1, HV] gate projections. A_log, dt_bias: [HV].
        recurrent_state: [B, HV, DK, DV], float32 or the activations' dtype, contiguous; advanced in place.
    Returns:
        out [B, 1, HV, DV] in the activations' dtype (the core attention output, before the gated RMSNorm).
    """
    dtype = conv_weight.dtype
    x = mixed_qkv.to(dtype)
    if x.stride(-1) != 1 or x.stride(0) % 4:
        x = x.contiguous()
    if (
        conv_state.dtype != dtype
        or not conv_state.is_contiguous()
        or not recurrent_state.is_contiguous()
        or recurrent_state.dtype not in (torch.float32, dtype)
    ):
        # Layouts the fused kernel does not take: the unfused path, same math
        qkv = causal_conv1d_update(x.transpose(1, 2), conv_state, conv_weight, conv_bias, activation).transpose(1, 2)
        bsz = x.shape[0]
        q, k, v = qkv.split([num_k_heads * head_k_dim, num_k_heads * head_k_dim, num_v_heads * head_v_dim], -1)
        q, k = q.reshape(bsz, 1, num_k_heads, head_k_dim), k.reshape(bsz, 1, num_k_heads, head_k_dim)
        g = -A_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias.float())
        out, final = gated_delta_rule(
            q, k, v.reshape(bsz, 1, num_v_heads, head_v_dim), g, b.sigmoid(), recurrent_state, True, use_qk_l2norm
        )
        if final is not recurrent_state:
            recurrent_state.copy_(final)
        return out
    args = (
        x, conv_state, conv_weight.contiguous(), None if conv_bias is None else conv_bias.to(dtype).contiguous(),
        a.to(dtype).contiguous(), b.to(dtype).contiguous(), A_log.float().contiguous(), dt_bias.float().contiguous(),
        recurrent_state, num_k_heads, head_k_dim, num_v_heads, head_v_dim, _silu(activation), int(use_qk_l2norm),
    )  # fmt: skip
    return (_gdn_decode_impl if _eager() else _gdn_decode)(*args)
