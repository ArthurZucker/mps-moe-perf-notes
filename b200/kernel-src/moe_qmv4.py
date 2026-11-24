"""Build (sm_100, local nvcc) and register the MoE 4-bit expert matvec CUDA kernels as torch custom ops."""
import os
import torch
from torch.utils.cpp_extension import load

_here = os.path.dirname(os.path.abspath(__file__))
_ext = load(
    name="moe_qmv4_ext",
    sources=[os.path.join(_here, "moe_qmv4.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17"],
    extra_cflags=["-O3"],
    verbose=False,
)


@torch.library.custom_op("moe_cuda::gather_qmv4", mutates_args=())
def gather_qmv4(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, b: torch.Tensor, ids: torch.Tensor, x_div: int, group_size: int) -> torch.Tensor:
    return _ext.gather_qmv4(x, w.view(torch.int32), s, b, ids, x_div, group_size)


@gather_qmv4.register_fake
def _(x, w, s, b, ids, x_div, group_size):
    return x.new_empty(ids.numel(), w.shape[1])


@torch.library.custom_op("moe_cuda::down_route4", mutates_args=())
def down_route4(act: torch.Tensor, w: torch.Tensor, s: torch.Tensor, b: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, topk: int, group_size: int) -> torch.Tensor:
    return _ext.down_route4(act, w.view(torch.int32), s, b, ids, weights, topk, group_size)


@down_route4.register_fake
def _(act, w, s, b, ids, weights, topk, group_size):
    return act.new_empty(ids.numel() // topk, w.shape[1])
