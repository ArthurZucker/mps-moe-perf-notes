"""Prefill MoE experts on Metal straight from NVFP4 / MXFP8 banks: pairs sorted by expert, two grouped GEMMs
(gate/up + activation, down * route weight), deterministic routing sum. Same arguments as fp4_experts_opt.experts_decode.

    experts_prefill(x [T, H], gu_weight, gu_scale, gu_factor, dn_weight, dn_scale, dn_factor, ids [T, R],
                    weights [T, R], gate_cap, up_cap) -> [T, H]
"""
import pathlib

import torch

_NAMES = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}
_WEIGHT_KINDS = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
_LIB = []


def _lib():
    if not _LIB:
        _LIB.append(torch.mps.compile_shader((pathlib.Path(__file__).parent / "fp4_gemm.metal").read_text()))
    return _LIB[0]


def _tiles(ids, experts, BM):
    """Sorted pair order and the (expert, first sorted row) of every BM-row tile, padded to a static upper bound."""
    flat = ids.reshape(-1)
    perm = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=experts)
    row_end = torch.cumsum(counts, 0)
    row_start = row_end - counts
    tiles = (counts + BM - 1) // BM
    tile_end = torch.cumsum(tiles, 0)
    max_tiles = (flat.numel() + BM - 1) // BM + experts
    t = torch.arange(max_tiles, device=ids.device)
    tile_e = torch.searchsorted(tile_end, t, right=True)
    valid = tile_e < experts
    e = tile_e.clamp(max=experts - 1)
    tile_row = row_start[e] + (t - (tile_end[e] - tiles[e])) * BM
    return perm, torch.where(valid, tile_e, -1).int(), tile_row.int(), row_end.int(), max_tiles


def experts_prefill(x, gu_weight, gu_scale, gu_factor, dn_weight, dn_scale, dn_factor, ids, weights, gate_cap, up_cap):
    fmt = 0 if gu_weight.dtype == torch.uint8 else 1
    tokens, routes = ids.shape
    experts, hidden, inter = gu_weight.shape[0], dn_weight.shape[1], gu_weight.shape[1] // 2
    k = hidden
    bm = 64 if tokens * routes >= 64 * experts else 32  # 64-row tiles once experts average 64+ rows
    perm, tile_e, tile_row, row_end, max_tiles = _tiles(ids, experts, bm)
    lib, name = _lib(), _NAMES[x.dtype]
    act = torch.empty(tokens * routes, inter, dtype=x.dtype, device=x.device)
    getattr(lib, f"gemm_gate_up_{fmt}_{name}_{bm}")(
        gu_weight.view(torch.uint8), gu_scale.view(torch.uint8), gu_factor.float(), x.contiguous(), perm, tile_e,
        tile_row, row_end, act, inter, k, routes, float(gate_cap), float(up_cap),
        threads=[max_tiles * 128, inter // 32], group_size=[128, 1],
    )  # fmt: skip
    y = torch.empty(tokens * routes, hidden, dtype=torch.float32, device=x.device)
    w = weights.contiguous()
    getattr(lib, f"gemm_down_{fmt}_{name}_{bm}")(
        dn_weight.view(torch.uint8), dn_scale.view(torch.uint8), dn_factor.float(), act, perm, w.view(torch.uint8),
        tile_e, tile_row, row_end, y, hidden, inter, _WEIGHT_KINDS[w.dtype] << 1,
        threads=[max_tiles * 128, hidden // 32], group_size=[128, 1],
    )  # fmt: skip
    out = torch.empty_like(y)
    out[perm] = y  # back to (token, route) order
    return out.view(tokens, routes, hidden).sum(1).to(x.dtype)
