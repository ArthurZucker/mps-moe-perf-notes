"""Prefill attention for Inkling-2 on Metal (see prefill_attention.metal): one dispatch, nothing [H, T, L] materialized."""
import pathlib

import torch

_NAMES = {torch.float16: "half", torch.bfloat16: "bfloat"}
_LIB = []
_PROJ = {}


def _lib():
    if not _LIB:
        _LIB.append(torch.mps.compile_shader((pathlib.Path(__file__).parent / "prefill_attention.metal").read_text()))
    return _LIB[0]


def _prefill_attention_mps(query, key, value, rel_states, proj, distance, sinks, scaling, window):
    """query [1, H, T, 128], key / value [1, Hkv, L, 128] (L % 32 == 0), rel_states [1, T, H, 16], proj [16, R],
    distance [T, L] int64 (affine: base + i - j), sinks [1, T, H], window (None = full) -> [1, T, H, 128]."""
    _, heads, tokens, dim = query.shape
    kv_heads, length = key.shape[1], key.shape[2]
    padded = -(-tokens // 32) * 32
    pad = padded - tokens

    def padq(x, d):  # pad the token dim
        return torch.nn.functional.pad(x, [0, 0] * (x.dim() - 1 - d) + [0, pad]) if pad else x

    q = padq(query[0], 1).contiguous()
    rs = padq(rel_states[0].float(), 0).contiguous()
    sk = padq(sinks[0].float(), 0).contiguous()
    proj_t = _PROJ.get(proj.data_ptr())
    if proj_t is None:
        proj_t = _PROJ[proj.data_ptr()] = proj.detach().float().t().contiguous()
    out = torch.empty(padded, heads, dim, dtype=query.dtype, device=query.device)
    getattr(_lib(), f"prefill_attention_{_NAMES[query.dtype]}")(
        q, key[0].contiguous(), value[0].contiguous(), rs, proj_t, sk, distance.contiguous(), out, heads,
        heads // kv_heads, tokens, padded, length, proj_t.shape[0], window if window > 0 else 1 << 30, float(scaling),
        threads=[heads * 128, padded // 32], group_size=[128, 1],
    )  # fmt: skip
    return out[:tokens].unsqueeze(0)


# A torch library op with a fake, so a compiled verify / prefill graph calls the kernel instead of tracing into it.
_OPS = torch.library.Library("metal_prefill_attention", "DEF")
_OPS.define(
    "prefill_attention(Tensor query, Tensor key, Tensor value, Tensor rel_states, Tensor proj, Tensor distance, "
    "Tensor sinks, float scaling, int window) -> Tensor"
)
_OPS.impl("prefill_attention", _prefill_attention_mps, "MPS")


@torch.library.register_fake("metal_prefill_attention::prefill_attention")
def _(query, key, value, rel_states, proj, distance, sinks, scaling, window):
    return query.new_empty(1, query.shape[2], query.shape[1], query.shape[3])


def prefill_attention(query, key, value, rel_states, proj, distance, sinks, scaling, window):
    """See `_prefill_attention_mps`; `window` None (or 0) = full attention."""
    return torch.ops.metal_prefill_attention.prefill_attention(
        query, key, value, rel_states, proj, distance, sinks, float(scaling), int(window or 0)
    )
