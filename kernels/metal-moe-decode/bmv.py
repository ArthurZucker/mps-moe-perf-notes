"""Batch-1 bf16 / fp16 matvecs for exact decode (see bmv.metal): `bmv_multi(x0, w0, x1, w1, x2, w2)` -> three outputs
in one dispatch, a torch library op with a fake (torch.compile friendly)."""
import pathlib

import torch

_NAMES = {torch.float16: "half", torch.bfloat16: "bfloat"}
_NSG = 8
_LIB = []


def _lib():
    if not _LIB:
        _LIB.append(torch.mps.compile_shader((pathlib.Path(__file__).parent / "bmv.metal").read_text()))
    return _LIB[0]


def _bmv_multi_mps(x0, w0, x1, w1, x2, w2):
    ns = [w.shape[0] for w in (w0, w1, w2)]
    tgs = [-(-n // _NSG) for n in ns]
    ys = [x.new_empty(*x.shape[:-1], n) for x, n in zip((x0, x1, x2), ns)]
    getattr(_lib(), f"bmv_multi_{_NAMES[x0.dtype]}")(
        w0, x0.contiguous(), ys[0], w1, x1.contiguous(), ys[1], w2, x2.contiguous(), ys[2], ns[0], ns[1], ns[2],
        tgs[0], tgs[1], w0.shape[1], threads=sum(tgs) * _NSG * 32, group_size=_NSG * 32,
    )  # fmt: skip
    return tuple(ys)


_OPS = torch.library.Library("metal_bmv", "DEF")
_OPS.define("bmv_multi(Tensor x0, Tensor w0, Tensor x1, Tensor w1, Tensor x2, Tensor w2) -> (Tensor, Tensor, Tensor)")
_OPS.impl("bmv_multi", _bmv_multi_mps, "MPS")


@torch.library.register_fake("metal_bmv::bmv_multi")
def _(x0, w0, x1, w1, x2, w2):
    return tuple(x.new_empty(*x.shape[:-1], w.shape[0]) for x, w in ((x0, w0), (x1, w1), (x2, w2)))


bmv_multi = torch.ops.metal_bmv.bmv_multi.default
