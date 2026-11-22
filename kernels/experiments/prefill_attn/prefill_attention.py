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


def prefill_attention(query, key, value, rel_states, proj, distance, sinks, scaling, window):
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
        heads // kv_heads, tokens, padded, length, proj_t.shape[0], window if window else 1 << 30, float(scaling),
        threads=[heads * 128, padded // 32], group_size=[128, 1],
    )  # fmt: skip
    return out[:tokens].unsqueeze(0)
