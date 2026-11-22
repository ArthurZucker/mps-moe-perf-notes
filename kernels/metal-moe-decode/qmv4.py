"""Batch-1 4-bit affine matvec (MLX packing) for Apple GPUs, tuned per decode shape class toward the read roofline.
qmv4.metal holds the kernel template and one instantiation per work layout; this module compiles it with
`torch.mps.compile_shader`, picks a layout per (N, K) and exposes `qmv4(x, weight, scales, biases, group_size)` as a
torch library op with a fake (torch.compile friendly)."""

import functools
import pathlib

import torch


# Work layouts (LPR, ROWS, VPL, SPLIT, NSG), see qmv4.metal (which instantiates exactly these); tuned with tune.py on
# an M3 Max (30-core GPU).
LAYOUTS = {
    # few rows, long K (324 x 10240): split-K over 8 simdgroups, otherwise too few loads in flight
    "thin": (16, 1, 16, 8, 8),
    # short K (<= 1024), many rows (10240 x 320): 8 lanes per row, 4 rows per simdgroup, one word per lane per step
    "short_k_tall": (8, 1, 8, 1, 4),
    # short K, moderate rows (2560 x 640): 2-way split-K keeps the threadgroups busy
    "short_k": (8, 1, 16, 2, 4),
    # many long rows (>= 6144 x 2304, lm_head): a full simdgroup on 2 rows, coalesced 1 KB per row per step
    "tall": (32, 2, 16, 1, 2),
    # everything in between (1281..2560 rows x 2304..6144)
    "mid": (8, 1, 16, 2, 2),
}

_NAMES = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader((pathlib.Path(__file__).parent / "qmv4.metal").read_text())
    return _LIB


def _layout(n, k):
    if n <= 512:
        return LAYOUTS["thin"]
    if k <= 1024:
        return LAYOUTS["short_k_tall" if n >= 8192 else "short_k"]
    return LAYOUTS["tall" if n >= 6144 else "mid"]


@functools.cache
def _launch(dtype, group_size, n, k):
    """Kernel and grid for one (dtype, group size, shape): resolved once, the decode loop only pays the dispatch."""
    lpr, rows, vpl, split, nsg = _layout(n, k)
    rows_per_tg = nsg // split * (32 // lpr) * rows
    kernel = getattr(_lib(), f"qmv4_{_NAMES[dtype]}_{group_size}_{lpr}_{rows}_{vpl}_{split}_{nsg}")
    return kernel, -(-n // rows_per_tg) * nsg * 32, nsg * 32


def _qmv4_mps(x, weight, scales, biases, group_size):
    n, words = weight.shape
    kernel, threads, group = _launch(x.dtype, group_size, n, words * 8)
    out = torch.empty(*x.shape[:-1], n, dtype=x.dtype, device=x.device)
    kernel(weight, scales, biases, x.contiguous(), out, n, words * 8, threads=threads, group_size=group)
    return out


MULTI_LAYOUT = LAYOUTS["tall"]


def _qmv4_multi_mps(x0, w0, s0, b0, x1, w1, s1, b1, x2, w2, s2, b2, group_size):
    """Three batch-1 matvecs (each its own x and weight, same K) in one dispatch -> (y0, y1, y2)."""
    lpr, rows, vpl, split, nsg = MULTI_LAYOUT
    rows_per_tg = nsg // split * (32 // lpr) * rows
    k = w0.shape[1] * 8
    ns = [w.shape[0] for w in (w0, w1, w2)]
    tgs = [-(-n // rows_per_tg) for n in ns]
    ys = [x.new_empty(*x.shape[:-1], n) for x, n in zip((x0, x1, x2), ns)]
    kernel = getattr(_lib(), f"qmv4_multi_{_NAMES[x0.dtype]}_{group_size}_{lpr}_{rows}_{vpl}_{split}_{nsg}")
    kernel(w0, s0, b0, x0.contiguous(), ys[0], w1, s1, b1, x1.contiguous(), ys[1], w2, s2, b2, x2.contiguous(), ys[2],
           ns[0], ns[1], ns[2], tgs[0], tgs[1], k, threads=sum(tgs) * nsg * 32, group_size=nsg * 32)  # fmt: skip
    return tuple(ys)


# Registered through the low-level Library API rather than `torch.library.custom_op`: the latter's Python dispatch
# layers cost ~4 us of host time per call, which at batch 1 is as long as the small matvecs themselves (2560 x 640:
# 17 -> 9 us end to end on an M3 Max).
_OPS = torch.library.Library("metal_qmv4", "DEF")
_OPS.define("qmv4(Tensor x, Tensor weight, Tensor scales, Tensor biases, int group_size) -> Tensor")
_OPS.impl("qmv4", _qmv4_mps, "MPS")


_OPS.define(
    "qmv4_multi(Tensor x0, Tensor w0, Tensor s0, Tensor b0, Tensor x1, Tensor w1, Tensor s1, Tensor b1, Tensor x2, "
    "Tensor w2, Tensor s2, Tensor b2, int group_size) -> (Tensor, Tensor, Tensor)"
)
_OPS.impl("qmv4_multi", _qmv4_multi_mps, "MPS")


@torch.library.register_fake("metal_qmv4::qmv4_multi")
def _(x0, w0, s0, b0, x1, w1, s1, b1, x2, w2, s2, b2, group_size):
    return tuple(x.new_empty(*x.shape[:-1], w.shape[0]) for x, w in ((x0, w0), (x1, w1), (x2, w2)))


@torch.library.register_fake("metal_qmv4::qmv4")
def _(x, weight, scales, biases, group_size):
    return x.new_empty(*x.shape[:-1], weight.shape[0])


qmv4 = torch.ops.metal_qmv4.qmv4.default
qmv4_multi = torch.ops.metal_qmv4.qmv4_multi.default
"""qmv4(x, weight, scales, biases, group_size): `x` [..., K] holding a single row, `weight` [N, K / 8] uint32 (MLX
4-bit packing), `scales` / `biases` [N, K / group_size] -> [..., N]."""
