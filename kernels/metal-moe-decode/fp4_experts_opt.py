"""MoE expert kernels on Metal straight from NVFP4 / MXFP8 checkpoints (no dequantized copy, no requantization).

Formats (ModelOpt / OCP, as stored by e.g. thinkingmachines/Inkling-2-Mini):
  NVFP4: weight uint8 [E, N, K / 2], two E2M1 codes per byte (even element in the low nibble), one float8 e4m3 scale per
         16 values [E, N, K / 16], a float32 global scale per expert (`factor`: [E, 2] for gate / up, [E] for down).
  MXFP8: weight float8 e4m3 [E, N, K], one e8m0 (power of two, uint8 biased by 127) scale per 32 values [E, N, K / 32].

Ops (namespace `fp4_experts_opt`), for T >= 1 tokens with R routes each (`ids` int32 / `weights` float32 [T, R]):
  gate_up_act(x, weight, scale, factor, ids, weights, gate_cap, up_cap) -> [T * R, I]
      act = softcap(g, gate_cap) * sigmoid(g) * softcap(u, up_cap) of the [gate; up] bank (gate_cap = 0: SiLU(g) * u)
  down_sum(act, weight, scale, factor, ids, weights) -> [T, H]
  experts_decode(x, gate_up..., down..., ids, weights, gate_cap, up_cap) -> [T, H]   (both, 2 dispatches)
  experts_fused(same arguments) -> [T, H]   (1 dispatch, fp32 atomics; see below)
`ids` may be int32 or int64 and `weights` float32, bfloat16 or float16: pass the router's top-k tensors as they are,
the kernels read them directly (the caller's `.int()` / `.float()` were two extra dispatches, ~6 us per evaluation).
Zero-weight routes read nothing.

`experts_fused` computes, per (64-wide intermediate slice, pair) threadgroup, the slice's activations and their down
contribution, accumulating into an fp32 workspace with atomics. At T = 1 with all routes active it is on par with the
two dispatches (0-5 us per evaluation in the 48-layer sequence, within noise); it is 2x slower on masked passes (few
threadgroups) and 1.3-1.7x slower at T = 8 (atomic traffic), and its output depends on the atomics' order, so
`experts_decode` keeps the two dispatches. With T > 1 (speculative verify) the grids run all tokens of a row block side by side,
so an expert shared by several tokens is read from DRAM once and from the cache after that.
"""

import pathlib

import torch


_NAMES = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}
_LIBS = {}

# Tiles per format (0: NVFP4, 1: MXFP8), tuned on an M3 Max (30-core GPU) at Inkling-2-Mini shapes, T = 1 and 8
# (bench.py --sweep). rows: outputs per simdgroup (gate/up) or per threadgroup (down); vpl: values per lane per step;
# sg: simdgroups per threadgroup.
TILES = {
    ("gate_up", 0): {"rows": 1, "vpl": 8},
    ("gate_up", 1): {"rows": 1, "vpl": 8},
    ("down", 0): {"rows": 4, "vpl": 16, "sg": 4},
    ("down", 1): {"rows": 2, "vpl": 16, "sg": 4},
}
FUSED_TILE = {"sg": 8, "vpl": 8}  # experts_decode: simdgroups per (pair, 64-wide slice) threadgroup
MAX_ROUTES = 64  # down_sum compacts a token's active routes in threadgroup memory
_WEIGHT_KINDS = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
_WORKSPACES = {}


def _routing(ids: torch.Tensor, weights: torch.Tensor):
    """`ids` / `weights` as the router produced them (int32 or int64, float32 / bfloat16 / float16): the kernels read
    them directly, so the caller needs no cast dispatches. Returns the tensors and the kernels' `mode`."""
    assert ids.dtype in (torch.int32, torch.int64) and weights.dtype in _WEIGHT_KINDS
    mode = int(ids.dtype == torch.int64) | _WEIGHT_KINDS[weights.dtype] << 1
    return ids.contiguous(), weights.contiguous(), mode


def _workspace(device: torch.device, size: int):
    """Zeroed fp32 accumulator (at least `size`) + ticket for experts_fused, which leaves both zeroed on exit."""
    key = str(device)
    if key not in _WORKSPACES or _WORKSPACES[key][0].numel() < size:
        _WORKSPACES[key] = (torch.zeros(size, device=device), torch.zeros(1, dtype=torch.int32, device=device))
    return _WORKSPACES[key]


def _lib(tune: bool = False):
    if tune not in _LIBS:
        source = (pathlib.Path(__file__).parent / "fp4_experts_opt.metal").read_text()
        _LIBS[tune] = torch.mps.compile_shader(("#define TUNE 1\n" if tune else "") + source)
    return _LIBS[tune]


def _fmt(weight: torch.Tensor) -> int:
    return 0 if weight.dtype == torch.uint8 else 1  # NVFP4 packs codes in uint8, MXFP8 stores float8 values


def launch_gate_up(x, weight, scale, factor, ids, weights, gate_cap, up_cap, tune=False, **tile):
    """`x` [T, K], gate/up bank [E, 2I, K / 2] uint8 (NVFP4) or [E, 2I, K] float8 (MXFP8) -> act [T * R, I]."""
    fmt, (tokens, routes) = _fmt(weight), ids.shape
    t = {**TILES["gate_up", fmt], **tile}
    inter, k = weight.shape[1] // 2, weight.shape[2] * (2 if fmt == 0 else 1)
    pairs = tokens * routes
    assert inter % t["rows"] == 0 and k % t["vpl"] == 0
    out = torch.empty(pairs, inter, dtype=x.dtype, device=x.device)
    threads = pairs * (inter // t["rows"]) * 32
    ids, weights, mode = _routing(ids, weights)
    getattr(_lib(tune), f"gate_up_act_{fmt}_{_NAMES[x.dtype]}_{t['rows']}_{t['vpl']}")(
        weight.view(torch.uint8), scale.view(torch.uint8), factor, x.contiguous(), ids, weights, out, inter, k, pairs,
        routes, mode, float(gate_cap), float(up_cap),
        threads=-(-threads // 64) * 64, group_size=64,
    )  # fmt: skip
    return out


def launch_down(act, weight, scale, factor, ids, weights, tune=False, **tile):
    """`act` [T * R, I], down bank [E, H, I / 2] uint8 or [E, H, I] float8 -> [T, H]."""
    t = {**TILES["down", _fmt(weight)], **tile}
    (tokens, routes), hidden, inter = ids.shape, weight.shape[1], act.shape[-1]
    assert routes <= MAX_ROUTES and hidden % t["rows"] == 0 and inter % t["vpl"] == 0
    out = torch.empty(tokens, hidden, dtype=act.dtype, device=act.device)
    name = f"down_sum_{_fmt(weight)}_{_NAMES[act.dtype]}_{t['rows']}_{t['vpl']}_{t['sg']}"
    ids, weights, mode = _routing(ids, weights)
    getattr(_lib(tune), name)(
        weight.view(torch.uint8), scale.view(torch.uint8), factor, act.contiguous(), ids, weights, out, hidden, inter,
        routes, tokens, mode,
        threads=hidden // t["rows"] * tokens * 32 * t["sg"], group_size=32 * t["sg"],
    )  # fmt: skip
    return out


def launch_fused(x, gu_weight, gu_scale, gu_factor, dn_weight, dn_scale, dn_factor, ids, weights, gate_cap, up_cap,
                 tune=False, **tile):  # fmt: skip
    """Single dispatch: `x` [T, H] -> [T, H]."""
    t = {**FUSED_TILE, **tile}
    fmt, (tokens, routes) = _fmt(gu_weight), ids.shape
    hidden, inter = dn_weight.shape[1], gu_weight.shape[1] // 2
    assert inter % 64 == 0 and _fmt(dn_weight) == fmt
    out = torch.empty(tokens, hidden, dtype=x.dtype, device=x.device)
    ws, ticket = _workspace(x.device, tokens * hidden)
    ids, weights, mode = _routing(ids, weights)
    pairs = tokens * routes
    getattr(_lib(tune), f"experts_fused_{fmt}_{_NAMES[x.dtype]}_{t['sg']}_{t['vpl']}")(
        gu_weight.view(torch.uint8), gu_scale.view(torch.uint8), gu_factor, dn_weight.view(torch.uint8),
        dn_scale.view(torch.uint8), dn_factor, x.contiguous(), ids, weights, out, ws, ticket, hidden, inter, pairs,
        routes, mode, float(gate_cap), float(up_cap),
        threads=pairs * (inter // 64) * 32 * t["sg"], group_size=32 * t["sg"],
    )  # fmt: skip
    return out


DOWN_ROUTE_TILE = {"lpr": 8, "rpl": 1, "vpl": 32, "sg": 4}  # one threadgroup per (pair, 32 rows): M3 Max, T = 1


def launch_down_route(act, weight, scale, factor, ids, weights):
    """`act` [T * R, I] -> per-pair down projections scaled by their route weights, fp32 [T * R, H]."""
    t = DOWN_ROUTE_TILE
    hidden, inter, pairs = weight.shape[1], act.shape[-1], act.shape[0]
    rows_tg = (32 // t["lpr"]) * t["rpl"] * t["sg"]
    assert hidden % rows_tg == 0 and inter % t["vpl"] == 0 and act.dtype in (torch.bfloat16, torch.float16)
    out = torch.empty(pairs, hidden, dtype=torch.float32, device=act.device)
    ids, weights, mode = _routing(ids, weights)
    getattr(_lib(), f"down_route_{_fmt(weight)}_{_NAMES[act.dtype]}")(
        weight.view(torch.uint8), scale.view(torch.uint8), factor, act.contiguous(), ids, weights, out, hidden, inter,
        mode, threads=pairs * (hidden // rows_tg) * 32 * t["sg"], group_size=32 * t["sg"],
    )  # fmt: skip
    return out


def _down_route(act, weight, scale, factor, ids, weights):
    return launch_down_route(act, weight, scale, factor, ids, weights)


def _gate_up_act(x, weight, scale, factor, ids, weights, gate_cap, up_cap):
    return launch_gate_up(x, weight, scale, factor, ids, weights, gate_cap, up_cap)


def _down_sum(act, weight, scale, factor, ids, weights):
    return launch_down(act, weight, scale, factor, ids, weights)


def _experts_decode(x, gu_weight, gu_scale, gu_factor, dn_weight, dn_scale, dn_factor, ids, weights, gate_cap, up_cap):
    act = launch_gate_up(x, gu_weight, gu_scale, gu_factor, ids, weights, gate_cap, up_cap)
    return launch_down(act, dn_weight, dn_scale, dn_factor, ids, weights)


def _experts_fused(x, gu_weight, gu_scale, gu_factor, dn_weight, dn_scale, dn_factor, ids, weights, gate_cap, up_cap):
    return launch_fused(x, gu_weight, gu_scale, gu_factor, dn_weight, dn_scale, dn_factor, ids, weights, gate_cap, up_cap)


_lib_def = torch.library.Library("fp4_experts_opt", "DEF")
_lib_def.define(
    "gate_up_act(Tensor x, Tensor weight, Tensor scale, Tensor factor, Tensor ids, Tensor weights, float gate_cap, "
    "float up_cap) -> Tensor"
)
_lib_def.define("down_sum(Tensor act, Tensor weight, Tensor scale, Tensor factor, Tensor ids, Tensor weights) -> Tensor")
_lib_def.define("down_route(Tensor act, Tensor weight, Tensor scale, Tensor factor, Tensor ids, Tensor weights) -> Tensor")
_lib_def.define(
    "experts_decode(Tensor x, Tensor gu_weight, Tensor gu_scale, Tensor gu_factor, Tensor dn_weight, Tensor dn_scale, "
    "Tensor dn_factor, Tensor ids, Tensor weights, float gate_cap, float up_cap) -> Tensor"
)
_lib_def.define(
    "experts_fused(Tensor x, Tensor gu_weight, Tensor gu_scale, Tensor gu_factor, Tensor dn_weight, Tensor dn_scale, "
    "Tensor dn_factor, Tensor ids, Tensor weights, float gate_cap, float up_cap) -> Tensor"
)
_lib_def.impl("experts_fused", _experts_fused, "MPS")
_lib_def.impl("gate_up_act", _gate_up_act, "MPS")
_lib_def.impl("down_sum", _down_sum, "MPS")
_lib_def.impl("down_route", _down_route, "MPS")
_lib_def.impl("experts_decode", _experts_decode, "MPS")


@torch.library.register_fake("fp4_experts_opt::gate_up_act")
def _(x, weight, scale, factor, ids, weights, gate_cap, up_cap):
    return x.new_empty(ids.numel(), weight.shape[1] // 2)


@torch.library.register_fake("fp4_experts_opt::down_route")
def _(act, weight, scale, factor, ids, weights):
    return act.new_empty(act.shape[0], weight.shape[1], dtype=torch.float32)


@torch.library.register_fake("fp4_experts_opt::down_sum")
def _(act, weight, scale, factor, ids, weights):
    return act.new_empty(ids.shape[0], weight.shape[1])


@torch.library.register_fake("fp4_experts_opt::experts_fused")
@torch.library.register_fake("fp4_experts_opt::experts_decode")
def _(x, gu_weight, gu_scale, gu_factor, dn_weight, dn_scale, dn_factor, ids, weights, gate_cap, up_cap):
    return x.new_empty(ids.shape[0], dn_weight.shape[1])


gate_up_act = torch.ops.fp4_experts_opt.gate_up_act
down_sum = torch.ops.fp4_experts_opt.down_sum
experts_decode = torch.ops.fp4_experts_opt.experts_decode
experts_fused = torch.ops.fp4_experts_opt.experts_fused
down_route = torch.ops.fp4_experts_opt.down_route
