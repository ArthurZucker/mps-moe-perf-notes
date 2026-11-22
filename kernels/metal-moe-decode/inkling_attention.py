"""Single-query (decode) attention for Inkling-2 on Apple GPUs.

Replaces, for one decoded token, the eager chain of `Inkling2RelativePositionBias` (projection, gather, masking) and
`eager_attention_forward` (repeat_kv, q.k, bias, mask, sink concat, fp32 softmax, p.v) with one Metal dispatch: a
split-K (flash-decoding) pass over (kv head, slot chunk) whose last threadgroup per kv head merges the chunk partials
and the sink. The relative-bias projection `relative_states @ proj` is fused: the [heads, kv_len] bias is never
materialized.

`inkling_attention.metal` holds the kernel; this module compiles it with `torch.mps.compile_shader` and exposes

    decode_attention(query, key_cache, value_cache, rel_states, rel_proj_t, key_distance, sinks, scaling)

as a torch library op with a fake, so torch.compile keeps it in graph.
"""

import functools
import pathlib

import torch

from .prefill_attention import prefill_attention


# Work layout, tuned with tune.py on an M3 Max (30-core GPU), bf16, 24 query / 8 kv heads: lanes per key, and per cache
# length the simdgroups per threadgroup and cache slots per threadgroup. 16-64 threadgroups of 128-512 threads read the
# cache fastest (one wave across the 30 cores).
LANES_PER_KEY = 16
MAX_CHUNKS = 64  # bound of the in-kernel merge, see MAX_CHUNKS in inkling_attention.metal


def _layout(kv_len, group):
    """(simdgroups per threadgroup, cache slots per threadgroup)."""
    if kv_len <= 512:
        nsg, chunk = 16, 256
    elif kv_len <= 2048:
        nsg, chunk = 8, -(-kv_len // 4)
    else:
        nsg, chunk = 4, max(512, -(-kv_len // MAX_CHUNKS))
    while nsg * group > 48:  # threadgroup memory: NSG * G * 128 floats <= 24 KB
        nsg //= 2
    return nsg, chunk


_NAMES = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}
_IDX = {torch.int32: "int", torch.int64: "long"}
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader((pathlib.Path(__file__).parent / "inkling_attention.metal").read_text())
    return _LIB


@functools.cache
def _kernel(dtype, idx_dtype, group, nsg, lpk):
    return getattr(_lib(), f"decode_attn_{_NAMES[dtype]}_{_IDX[idx_dtype]}_{group}_{nsg}_{lpk}")


@functools.cache
def _workspace(device, num_heads, n_chunks, head_dim):
    """Chunk partials and per-kv-head tickets, allocated once per shape. Calls on one device run in stream order and
    the kernel leaves the tickets at zero, so every call can reuse them."""
    part_o = torch.empty(num_heads, n_chunks, head_dim, dtype=torch.float32, device=device)
    part_ml = torch.empty(num_heads, n_chunks, 2, dtype=torch.float32, device=device)
    counters = torch.zeros(num_heads, dtype=torch.int32, device=device)
    return part_o, part_ml, counters


def _decode_attention_mps(query, key_cache, value_cache, rel_states, rel_proj_t, key_distance, sinks, scaling):
    num_heads, head_dim = query.shape[-3], query.shape[-1]
    num_kv_heads, kv_len = key_cache.shape[-3], key_cache.shape[-2]
    group = num_heads // num_kv_heads
    nsg, chunk = _layout(kv_len, group)
    n_chunks = -(-kv_len // chunk)
    kernel = _kernel(query.dtype, key_distance.dtype, group, nsg, LANES_PER_KEY)
    part_o, part_ml, counters = _workspace(query.device, num_heads, n_chunks, head_dim)
    out = torch.empty_like(query)
    kernel(
        query.contiguous(), key_cache.contiguous(), value_cache.contiguous(), rel_states.contiguous(),
        rel_proj_t.contiguous(), key_distance.contiguous(), sinks.contiguous(), out, part_o, part_ml, counters,
        kv_len, rel_proj_t.shape[0], chunk, scaling, threads=num_kv_heads * n_chunks * nsg * 32, group_size=nsg * 32,
    )  # fmt: skip
    return out


def _check(query, key_cache, value_cache, rel_states, rel_proj_t, key_distance, sinks):
    num_heads, head_dim = query.shape[-3], query.shape[-1]
    num_kv_heads, kv_len = key_cache.shape[-3], key_cache.shape[-2]
    torch._check(query.numel() == num_heads * head_dim, lambda: "decode_attention takes a single query token, batch 1")
    torch._check(head_dim == 128 and key_cache.shape[-1] == 128, lambda: "head_dim must be 128")
    torch._check(key_cache.numel() == num_kv_heads * kv_len * head_dim, lambda: "key_cache must be [1, Hkv, L, 128]")
    torch._check(value_cache.shape == key_cache.shape, lambda: "value_cache must match key_cache")
    torch._check(
        num_heads % num_kv_heads == 0 and num_heads // num_kv_heads in (1, 2, 3, 4, 6),
        lambda: "num_heads / num_kv_heads must be 1, 2, 3, 4 or 6",
    )
    torch._check(rel_states.numel() == num_heads * 16 and rel_proj_t.shape[-1] == 16, lambda: "d_rel must be 16")
    torch._check(key_distance.numel() == kv_len, lambda: "key_distance must hold one entry per cache slot")
    torch._check(sinks.numel() == num_heads, lambda: "sinks must hold one logit per query head")


# Registered through the low-level Library API rather than `torch.library.custom_op`, whose Python dispatch layers add
# ~4 us of host time per call.
_OPS = torch.library.Library("inkling_attention_opt", "DEF")
_OPS.define(
    "decode_attention(Tensor query, Tensor key_cache, Tensor value_cache, Tensor rel_states, Tensor rel_proj_t, "
    "Tensor key_distance, Tensor sinks, float scaling) -> Tensor"
)
_OPS.impl("decode_attention", _decode_attention_mps, "MPS")


@torch.library.register_fake("inkling_attention_opt::decode_attention")
def _(query, key_cache, value_cache, rel_states, rel_proj_t, key_distance, sinks, scaling):
    _check(query, key_cache, value_cache, rel_states, rel_proj_t, key_distance, sinks)
    return torch.empty_like(query)


decode_attention = torch.ops.inkling_attention_opt.decode_attention.default
"""decode_attention(query, key_cache, value_cache, rel_states, rel_proj_t, key_distance, sinks, scaling) -> out

Batch 1, one query token. All floating tensors share one dtype (float32, float16 or bfloat16).

- query        [1, H, 1, 128] (or any [..., H, 1|.., 128] view holding H * 128 contiguous values)
- key_cache    [1, Hkv, L, 128], value_cache likewise (H % Hkv == 0, query head h reads kv head h // (H / Hkv))
- rel_states   [..., H, 16]: `Inkling2RelativePositionBias` relative states of this token, after its norm
- rel_proj_t   [R, 16]: `proj.T.contiguous()` of the same module (R = rel_extent), prepare once at load time
- key_distance [L] int32 or int64: query position - key position per slot; negative marks an empty slot (masked)
- sinks        [..., H]: per-head sink logit
- scaling      float, head_dim ** -0.5

Returns a tensor shaped like `query`. Bias is rel_states[h] . rel_proj_t[d] for 0 <= d < R and zero beyond.
"""


def key_distance_for(relative_distance: torch.Tensor, kv_len: int) -> torch.Tensor:
    """`key_distance` from the `(1, kv_length)` `relative_distance` the model builds with `get_relative_distance`.

    The model's row has `kv_length` entries, aligned with the first slots of the returned cache. When it is shorter
    than the cache (a static sliding cache that is not full yet), the remaining slots are empty and marked -1."""
    row = relative_distance.reshape(-1)
    if row.numel() == kv_len:
        return row
    return torch.nn.functional.pad(row, (0, kv_len - row.numel()), value=-1)


def inkling2_attention(
    query, key, value, module, hidden_states, attention_mask, relative_distance, relative_states, attention_interface,
    **kwargs,
):
    """transformers' `inkling2_attention` on MPS: one decode token (batch 1, inference) runs the fused kernel (relative
    position bias + sinks + GQA in one dispatch); anything else takes the model's own path."""
    bias = module.relative_position_bias
    if query.shape[0] == 1 and query.shape[2] == 1 and not module.training:
        proj_t = getattr(bias, "proj_t", None)  # prepared once at load time when the caller sets it
        if proj_t is None or proj_t.dtype != query.dtype:
            proj_t = bias.proj.t().to(query.dtype).contiguous()
        sinks = module.sink_proj(hidden_states).to(query.dtype)
        distance = key_distance_for(relative_distance, key.shape[2])
        states = relative_states.to(query.dtype)
        out = decode_attention(query, key, value, states, proj_t, distance, sinks, module.scaling)
        return out.transpose(1, 2), None
    if (
        query.shape[0] == 1
        and not module.training
        and query.shape[-1] == 128
        and relative_states.shape[-1] == 16
        and query.dtype in (torch.bfloat16, torch.float16)
        and key.dtype == query.dtype
    ):  # prefill: flash attention with the bias, mask and sinks computed in-kernel
        length = key.shape[2]
        if length % 32:
            key, value = (torch.nn.functional.pad(x, (0, 0, 0, 32 - length % 32)) for x in (key, value))
            relative_distance = torch.nn.functional.pad(relative_distance, (0, 32 - length % 32), value=-1)
        sinks = module.sink_proj(hidden_states)
        out = prefill_attention(
            query, key, value, relative_states, bias.proj, relative_distance, sinks, module.scaling,
            module.sliding_window,
        )  # fmt: skip
        return out, None
    position_bias = bias.bias(relative_states, relative_distance)
    sinks = module.sink_proj(hidden_states).transpose(1, 2).unsqueeze(-1)
    dropout = module.attention_dropout if module.training else 0.0
    return attention_interface(
        module, query, key, value, attention_mask, dropout=dropout, scaling=module.scaling,
        sliding_window=module.sliding_window, position_bias=position_bias, sinks=sinks, **kwargs,
    )  # fmt: skip
