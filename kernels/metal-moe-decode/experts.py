"""Batch-1 MoE expert decode on Apple Silicon (Metal) for MLX affine 4-bit expert banks.

One token, R top-k routes:

    experts_decode(x, gate_up, gs, gb, down, ds, db, ids, weights, group_size) -> [1, H]
        = sum_r weights[r] * down[ids[r]] @ (SiLU(gate[ids[r]] @ x) * (up[ids[r]] @ x))

runs as two dispatches: `experts_gate_up` (gate and up rows of a route in the same simdgroup, SiLU(gate) * up applied
in registers, fp32 [R, I] product) then `experts_down_sum` (down projections + routing-weighted sum). The pieces are
exposed as ops too: `experts_gate_up_swiglu`, `experts_gate_up_masked` (raw [R, 2I] gate | up rows), `experts_down_sum`.

`launch_decode_fused` is a single-dispatch variant (fp32 atomics into a workspace): it ties the two-dispatch path with
all routes active but is ~2x slower on masked passes (a lone route only fills I / 64 threadgroups), and its output
depends on the atomics' order, so it is not the default. Routes whose weight is 0 read nothing in every op, so a correction
pass over 1-2 streamed routes costs only those routes.

Banks: `gate_up` [E, 2I, H / 8] uint32 (gate rows then up rows), `down` [E, H, I / 8], scales / biases
[E, N, K / group_size] in the activation dtype (float32, float16 or bfloat16), group_size 32, 64 or 128.
`ids` int32 [R], `weights` float32 [R].
"""

import pathlib

import torch


_NAMES = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}
_LIBS = {}
_WORKSPACES = {}

# Tile shapes, tuned on an M3 Max (30-core GPU) at Qwen3.8-Flash and Inkling-2-Mini expert shapes (bench.py --sweep).
#   rows: gate rows (each with its up row) or output rows per simdgroup / threadgroup
#   vpl: activations per lane per step (8: one uint32 per row, 16: a uint2, 32: a uint4)
#   slice: intermediate columns per threadgroup of the single-dispatch kernel
GATE_UP_TILE = {"rows": 1, "vpl": 16, "simdgroups": 2}
DOWN_TILE = {"rows": 2, "vpl": 16, "simdgroups": 4}
FUSED_TILE = {"slice": 64, "vpl": 16, "simdgroups": 8}
MAX_ROUTES = 64  # experts_down_sum compacts the active routes in threadgroup memory


def _lib(tune: bool = False):
    """The compiled kernels; `tune` adds the tile-sweep instantiations used by bench.py --sweep."""
    if tune not in _LIBS:
        source = (pathlib.Path(__file__).parent / "experts.metal").read_text()
        _LIBS[tune] = torch.mps.compile_shader(("#define EXPERTS_TUNE 1\n" if tune else "") + source)
    return _LIBS[tune]


def _vpl(k: int, vpl: int) -> int:
    """The widest load (up to `vpl` values per lane) whose chunks split a row of `k` evenly over a simdgroup."""
    while vpl > 8 and (k // vpl) % 32:
        vpl //= 2
    return vpl


def _workspace(device: torch.device, hidden: int):
    """Zeroed fp32 accumulator [H] + ticket for the single-dispatch kernel, which leaves both zeroed when it exits
    (dispatches on the MPS queue are serialized, so consecutive calls never share it concurrently)."""
    key = (str(device), hidden)
    if key not in _WORKSPACES:
        _WORKSPACES[key] = (torch.zeros(hidden, device=device), torch.zeros(1, dtype=torch.int32, device=device))
    return _WORKSPACES[key]


def launch_decode_fused(x, gate_up, gs, gb, down, ds, db, ids, weights, group_size, tune=False, **tile):
    """`x` [1, H] contiguous -> [1, H] in x's dtype, one dispatch (experimental, see the module docstring)."""
    tile = {**FUSED_TILE, **tile}
    hidden, inter, routes = down.shape[-2], down.shape[-1] * 8, ids.numel()
    slice_, sgs, vpl = tile["slice"], tile["simdgroups"], _vpl(hidden, tile["vpl"])
    assert inter % slice_ == 0, f"intermediate size {inter} must be a multiple of {slice_}"
    out = torch.empty(1, hidden, dtype=x.dtype, device=x.device)
    ws, ticket = _workspace(x.device, hidden)
    name = f"experts_decode_fused_{_NAMES[x.dtype]}_{group_size}_{slice_}_{sgs}_{vpl}"
    getattr(_lib(tune), name)(
        gate_up, gs, gb, down, ds, db, x, ids, weights, out, ws, ticket, hidden, inter, routes,
        threads=routes * (inter // slice_) * 32 * sgs, group_size=32 * sgs,
    )  # fmt: skip
    return out


def launch_gate_up(
    x, weight, scales, biases, ids, weights, group_size, swiglu=True, out_dtype=torch.float32, tune=False, **tile
):
    """`x` [1, K] contiguous, `weight` [E, 2I, K / 8] -> [R, I] SiLU(gate) * up (`swiglu`) or [R, 2I] raw rows."""
    tile = {**GATE_UP_TILE, **tile}
    routes, (rows2, words) = ids.numel(), weight.shape[-2:]
    inter, k = rows2 // 2, words * 8
    rows, vpl = tile["rows"], tile["vpl"] if tune else _vpl(k, tile["vpl"])
    out = torch.empty(routes, inter if swiglu else rows2, dtype=out_dtype, device=x.device)
    to = "f32" if out_dtype == torch.float32 and x.dtype != torch.float32 else "act"
    name = f"experts_gate_up_{'swiglu' if swiglu else 'raw'}_{_NAMES[x.dtype]}_{to}_{group_size}_{rows}_{vpl}"
    group = 32 * tile["simdgroups"]
    threads = routes * (inter // rows) * 32
    getattr(_lib(tune), name)(
        weight, scales, biases, x, ids, weights, out, inter, k, routes,
        threads=-(-threads // group) * group, group_size=group,
    )  # fmt: skip
    return out


def launch_down_sum(act, down, scales, biases, ids, weights, group_size, out_dtype=None, tune=False, **tile):
    """`act` [R, I] contiguous (fp32 or the weights' dtype), `down` [E, H, I / 8] -> [1, H] in `out_dtype`."""
    tile = {**DOWN_TILE, **tile}
    hidden, inter, routes = down.shape[-2], act.shape[-1], ids.numel()
    out_dtype = out_dtype or scales.dtype
    rows, sgs, vpl = tile["rows"], tile["simdgroups"], tile["vpl"]
    # the (route, chunk) pairs are flattened, so any chunk width dividing I keeps every thread busy
    assert routes <= MAX_ROUTES and hidden % rows == 0 and inter % vpl == 0 and group_size % vpl == 0
    out = torch.empty(1, hidden, dtype=out_dtype, device=act.device)
    ta = "f32" if act.dtype == torch.float32 and out_dtype != torch.float32 else "act"
    name = f"experts_down_sum_{_NAMES[out_dtype]}_{ta}_{group_size}_{rows}_{vpl}_{sgs}"
    getattr(_lib(tune), name)(
        down, scales, biases, act, ids, weights, out, hidden, inter, routes,
        threads=hidden // rows * 32 * sgs, group_size=32 * sgs,
    )  # fmt: skip
    return out


@torch.library.custom_op("metal_experts::experts_decode", mutates_args=())
def experts_decode(
    x: torch.Tensor, gate_up: torch.Tensor, gate_up_scales: torch.Tensor, gate_up_biases: torch.Tensor,
    down: torch.Tensor, down_scales: torch.Tensor, down_biases: torch.Tensor, ids: torch.Tensor,
    weights: torch.Tensor, group_size: int,
) -> torch.Tensor:  # fmt: skip
    """One token through its routed SwiGLU experts and their routing-weighted sum: `x` [1, H] -> [1, H]."""
    act = launch_gate_up(x.contiguous(), gate_up, gate_up_scales, gate_up_biases, ids, weights, group_size)
    return launch_down_sum(act, down, down_scales, down_biases, ids, weights, group_size, out_dtype=x.dtype)


@experts_decode.register_fake
def _(x, gate_up, gate_up_scales, gate_up_biases, down, down_scales, down_biases, ids, weights, group_size):
    return x.new_empty(1, down.shape[-2])


@torch.library.custom_op("metal_experts::experts_gate_up_swiglu", mutates_args=())
def experts_gate_up_swiglu(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, ids: torch.Tensor,
    weights: torch.Tensor, group_size: int,
) -> torch.Tensor:  # fmt: skip
    """SiLU(gate) * up of the routes with a nonzero weight (zeros for the others): `x` [1, K] -> [R, I] in x's dtype."""
    return launch_gate_up(x.contiguous(), weight, scales, biases, ids, weights, group_size, out_dtype=x.dtype)


@experts_gate_up_swiglu.register_fake
def _(x, weight, scales, biases, ids, weights, group_size):
    return x.new_empty(ids.numel(), weight.shape[-2] // 2)


@torch.library.custom_op("metal_experts::experts_gate_up_masked", mutates_args=())
def experts_gate_up_masked(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, ids: torch.Tensor,
    weights: torch.Tensor, group_size: int,
) -> torch.Tensor:  # fmt: skip
    """Raw gate | up rows of the routes with a nonzero weight (zeros for the others): `x` [1, K] -> [R, 2I]."""
    return launch_gate_up(x.contiguous(), weight, scales, biases, ids, weights, group_size, False, x.dtype)


@experts_gate_up_masked.register_fake
def _(x, weight, scales, biases, ids, weights, group_size):
    return x.new_empty(ids.numel(), weight.shape[-2])


@torch.library.custom_op("metal_experts::experts_down_sum", mutates_args=())
def experts_down_sum(
    act: torch.Tensor, down: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, ids: torch.Tensor,
    weights: torch.Tensor, group_size: int,
) -> torch.Tensor:  # fmt: skip
    """Routed down projections and their routing-weighted sum: `act` [R, I] -> [1, H] in act's dtype."""
    return launch_down_sum(act.contiguous(), down, scales, biases, ids, weights, group_size, out_dtype=act.dtype)


@experts_down_sum.register_fake
def _(act, down, scales, biases, ids, weights, group_size):
    return act.new_empty(1, down.shape[-2])
