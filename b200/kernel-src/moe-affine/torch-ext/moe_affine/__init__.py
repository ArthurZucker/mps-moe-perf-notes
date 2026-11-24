"""Affine 4-bit MoE expert matvecs (MetalExperts / MLX layout: codes LSB-first in uint32 words, per-group scales and
biases) for small-batch decode on CUDA."""
import torch

from ._ops import add_op_namespace_prefix, ops


def gather_qmv4(x, w, s, b, ids, x_div: int, group_size: int):
    """`out[r] = x[r // x_div] @ dequantize(w[ids[r]]).T` -> `[len(ids), N]` (bf16)."""
    return ops.gather_qmv4(x, w.view(torch.int32), s, b, ids, x_div, group_size)


def down_route4(act, w, s, b, ids, weights, top_k: int, group_size: int):
    """`out[t] = sum_j weights[t*top_k+j] * act[t*top_k+j] @ dequantize(w[ids[t*top_k+j]]).T` -> `[T, N]`; a route
    with weight 0 reads nothing."""
    return ops.down_route4(act, w.view(torch.int32), s, b, ids, weights, top_k, group_size)


@torch.library.register_fake(add_op_namespace_prefix("gather_qmv4"))
def _(x, w, s, b, ids, x_div, group_size):
    return x.new_empty(ids.numel(), w.shape[1])


@torch.library.register_fake(add_op_namespace_prefix("down_route4"))
def _(act, w, s, b, ids, weights, top_k, group_size):
    return act.new_empty(ids.numel() // top_k, w.shape[1])


def gemv_bf16(x, weight, bias=None):
    """`x [..., K] @ weight[N, K].T (+ bias)` for at most 8 rows of `x` (bf16)."""
    return ops.gemv_bf16(x, weight, bias)


@torch.library.register_fake(add_op_namespace_prefix("gemv_bf16"))
def _(x, weight, bias):
    return x.new_empty(*x.shape[:-1], weight.shape[0])


def offload_plan(ids, layer: int, num_experts: int, table, owner, age, clock):
    """Map one token's routed experts of `layer` to pool slots (LRU victims for misses; updates table / owner / age /
    clock in place) -> `(slots, miss)` int32."""
    return ops.offload_plan(ids, layer, num_experts, table, owner, age, clock)


def qmv4_offload(x, pool, host, ids, slots, miss, weights, top_k: int, x_div: int, group_size: int):
    """`pool`: `(w, scales, biases)` slot-indexed device banks; `host`: the data pointers of the expert-indexed banks
    in pinned host memory, same row layout (UVA). The pool is written (installs) without being declared mutated: the
    ops that read it are ordered by data flow, and declaring it would make compile copy it. Without `weights`: `[R, N]` per-route
    outputs (gate/up); with: the routing-weighted sum over each `top_k` routes (down). Misses read the host bank and
    install into their slot."""
    pw, ps, pb = pool
    hw, hs, hb = host
    return ops.qmv4_offload(x, pw.view(torch.int32), ps, pb, hw, hs, hb, ids, slots, miss, weights,
                            top_k, x_div, group_size)


@torch.library.register_fake(add_op_namespace_prefix("offload_plan"))
def _(ids, layer, num_experts, table, owner, age, clock):
    return torch.empty_like(ids), torch.empty_like(ids)


@torch.library.register_fake(add_op_namespace_prefix("qmv4_offload"))
def _(x, pw, ps, pb, hw, hs, hb, ids, slots, miss, weights, J, x_div, group_size):
    return x.new_empty(ids.numel() // J, pw.shape[1])


def offload_install(ids, slots, miss, host, pool, blocks: int = 16):
    """Copy the missed experts (`miss`) from the host banks (data pointers, UVA) into their pool `slots`."""
    ops.offload_install(ids, slots, miss, list(host), list(pool), blocks)


def qmv4_v2(x, pool, ids, top_k: int, x_div: int, group_size: int, host=None, slots=None, miss=None, weights=None, rows: int = 4):
    """v2 expert matvec. Resident: `pool` is the expert bank, `ids` index it. Streamed: `host` the data pointers of the
    host banks, `slots` / `miss` from `offload_plan` (misses read the host and install into their slot)."""
    pw, ps, pb = pool
    hw, hs, hb = host if host is not None else (0, 0, 0)
    return ops.qmv4_offload_v2(x, pw.view(torch.int32), ps, pb, hw, hs, hb, ids, slots, miss, weights, top_k, x_div,
                               group_size, rows)


@torch.library.register_fake(add_op_namespace_prefix("qmv4_offload_v2"))
def _(x, pw, ps, pb, hw, hs, hb, ids, slots, miss, weights, J, x_div, group_size, rows):
    return x.new_empty(ids.numel() // J, pw.shape[1])


def offload_lookup(ids, layer: int, num_experts: int, table, age, clock, weights):
    """Lookup only: `(slots, gpu_weights)`; missed routes get slot 0 and weight 0 (the host tier computes them)."""
    return ops.offload_lookup(ids, layer, num_experts, table, age, clock, weights)


def offload_lookup2(ids, layer: int, num_experts: int, table, age, clock, weights):
    """`offload_lookup` with a per-token clock (layer 0 advances it): ages are token stamps for `refresh_plan`."""
    return ops.offload_lookup2(ids, layer, num_experts, table, age, clock, weights)


def refresh_plan(ids, layer: int, num_experts: int, table, owner, age, clock, max_installs: int):
    """Slots for (up to `max_installs`) missed experts of `layer`, evicting only slots no later layer of this token reads.
    -> `(slots, miss)`: install the `miss` routes with `offload_install`."""
    return ops.refresh_plan(ids, layer, num_experts, table, owner, age, clock, max_installs)


@torch.library.register_fake(add_op_namespace_prefix("offload_lookup2"))
def _(ids, layer, num_experts, table, age, clock, weights):
    return torch.empty_like(ids), torch.empty_like(weights)


@torch.library.register_fake(add_op_namespace_prefix("refresh_plan"))
def _(ids, layer, num_experts, table, owner, age, clock, max_installs):
    return torch.empty_like(ids), torch.empty_like(ids)


def moe_layer_host(x, ids, weights, layer: int, num_experts: int, table, age, clock, pool, mailbox: int, result: int,
                   state, group_size: int, blocks: int = 148, post_inside: bool = True):
    """One token's routed experts of `layer` in one cooperative launch: lookup, host-tier post of the misses
    (`mailbox` 0: none), gate/up and down of the resident routes, host result added. `pool`: the 6 slot banks."""
    return ops.moe_layer_host(x, ids, weights, layer, num_experts, table, age, clock, list(pool), mailbox, result, state,
                              group_size, blocks, post_inside)


@torch.library.register_fake(add_op_namespace_prefix("moe_layer_host"))
def _(x, ids, weights, layer, num_experts, table, age, clock, pool, mailbox, result, state, group_size, blocks, post_inside):
    return x.new_empty(1, x.numel())


def host_post(ids, weights, layer: int, num_experts: int, table, x, mailbox: int, state):
    """Post the routes whose expert is not resident, with the activation `x`, to the host tier (mapped mailbox)."""
    ops.host_post(ids, weights, layer, num_experts, table, x, mailbox, state)


def host_wait(out, result: int, state):
    """`out` + the host tier's result for the last post (`out` itself when nothing was posted)."""
    return ops.host_wait(out, result, state)


def host_tier_start(banks, hidden: int, intermediate: int, group_size: int, cpus):
    """Start the CPU workers over the host banks (6 pointers per layer) -> `(handle, mailbox, result)`."""
    return ops.host_tier_start(list(banks), hidden, intermediate, group_size, list(cpus))


def host_tier_stop(handle: int):
    ops.host_tier_stop(handle)


def host_tier_timing(handle: int):
    """(seen->gate/up done ns, gate/up->down done ns, calls) summed since the last call (worker 0's clock)."""
    return ops.host_tier_timing(handle)


def host_tier_calls(handle: int) -> int:
    return ops.host_tier_calls(handle)


@torch.library.register_fake(add_op_namespace_prefix("offload_lookup"))
def _(ids, layer, num_experts, table, age, clock, weights):
    return torch.empty_like(ids), torch.empty_like(weights)


@torch.library.register_fake(add_op_namespace_prefix("host_wait"))
def _(out, result, state):
    return torch.empty_like(out)


def router_topk(x, weight, top_k: int, norm: bool):
    """One token's router: `(logits [1, E] bf16, weights [1, k] bf16, indices [1, k] int64)` of
    `topk(softmax(float(bf16(x @ weight.T))))`, renormalized when `norm`."""
    return ops.router_topk(x, weight, top_k, norm)


@torch.library.register_fake(add_op_namespace_prefix("router_topk"))
def _(x, weight, top_k, norm):
    t = x.numel() // weight.shape[1]
    return x.new_empty(t, weight.shape[0]), x.new_empty(t, top_k), x.new_empty(t, top_k, dtype=torch.long)


__all__ = ["gather_qmv4", "down_route4", "gemv_bf16", "offload_plan", "qmv4_offload", "offload_install", "router_topk", "qmv4_v2", "offload_lookup", "host_post", "host_wait", "host_tier_start", "host_tier_stop", "host_tier_calls", "offload_lookup2", "refresh_plan", "moe_layer_host"]


def gather_qmv4_v3(x, w, s, b, ids, x_div: int, group_size: int, rows: int = 4):
    """`gather_qmv4` with x staged in shared memory (v3)."""
    return ops.gather_qmv4_v3(x, w.view(torch.int32), s, b, ids, x_div, group_size, rows)


def down_route4_v3(act, w, s, b, ids, weights, top_k: int, group_size: int, rows: int = 2):
    """`down_route4` with the routes' activations staged in shared memory (v3)."""
    return ops.down_route4_v3(act, w.view(torch.int32), s, b, ids, weights, top_k, group_size, rows)


@torch.library.register_fake(add_op_namespace_prefix("gather_qmv4_v3"))
def _(x, w, s, b, ids, x_div, group_size, rows):
    return x.new_empty(ids.numel(), w.shape[1])


@torch.library.register_fake(add_op_namespace_prefix("down_route4_v3"))
def _(act, w, s, b, ids, weights, top_k, group_size, rows):
    return act.new_empty(ids.numel() // top_k, w.shape[1])


def gather_qmv4_mma(x, w, s, b, ids, x_div: int, group_size: int, rt: int = 2):
    """`gather_qmv4` on tensor cores (group size 64)."""
    return ops.gather_qmv4_mma(x, w.view(torch.int32), s, b, ids, x_div, group_size, rt)


def down_route4_mma(act, w, s, b, ids, weights, top_k: int, group_size: int, rt: int = 1):
    """`down_route4` on tensor cores (group size 64)."""
    return ops.down_route4_mma(act, w.view(torch.int32), s, b, ids, weights, top_k, group_size, rt)


@torch.library.register_fake(add_op_namespace_prefix("gather_qmv4_mma"))
def _(x, w, s, b, ids, x_div, group_size, rt):
    return x.new_empty(ids.numel(), w.shape[1])


@torch.library.register_fake(add_op_namespace_prefix("down_route4_mma"))
def _(act, w, s, b, ids, weights, top_k, group_size, rt):
    return act.new_empty(ids.numel() // top_k, w.shape[1])


def gather_qmv4_tc(x, w, s, b, ids, x_div: int, group_size: int):
    """`gather_qmv4` on tensor cores, row tiles staged with cp.async (group size 64)."""
    return ops.gather_qmv4_tc(x, w.view(torch.int32), s, b, ids, x_div, group_size)


def down_route4_tc(act, w, s, b, ids, weights, top_k: int, group_size: int):
    """`down_route4` on tensor cores, row tiles staged with cp.async (group size 64)."""
    return ops.down_route4_tc(act, w.view(torch.int32), s, b, ids, weights, top_k, group_size)


@torch.library.register_fake(add_op_namespace_prefix("gather_qmv4_tc"))
def _(x, w, s, b, ids, x_div, group_size):
    return x.new_empty(ids.numel(), w.shape[1])


@torch.library.register_fake(add_op_namespace_prefix("down_route4_tc"))
def _(act, w, s, b, ids, weights, top_k, group_size):
    return act.new_empty(ids.numel() // top_k, w.shape[1])


def tc6_supported(in_features: int, top_k_sum: int, group_size: int) -> bool:
    """Shapes the v6 tensor-core kernel covers: gate/up (K=2560, one route per output) and down (K=640, <= 10 routes)."""
    return group_size == 64 and ((in_features == 2560 and top_k_sum == 1) or (in_features == 640 and top_k_sum <= 10))


def qmv4_tc6(x, pool, ids, top_k: int, x_div: int, group_size: int, host=None, slots=None, miss=None, weights=None):
    """v6 tensor-core expert matvec. Resident: `pool` is the expert-indexed bank and `ids` index it. Streamed: `pool` is
    the slot-indexed pool, `host` the data pointers of the expert-indexed host banks, `slots` / `miss` from
    `offload_plan`: misses read the host banks and their tiles are written into the slot."""
    pw, ps, pb = pool
    hw, hs, hb = host if host is not None else (0, 0, 0)
    return ops.qmv4_tc6(x, pw.view(torch.int32), ps, pb, hw, hs, hb, ids, slots, miss, weights, top_k, x_div, group_size)


@torch.library.register_fake(add_op_namespace_prefix("qmv4_tc6"))
def _(x, pw, ps, pb, hw, hs, hb, ids, slots, miss, weights, J, x_div, group_size):
    return x.new_empty(ids.numel() // J, pw.shape[1])


def host_post2(ids, weights, layer: int, gpu_weights, x, mailbox: int, state):
    """Host tier v2: post the routes the lookup left to the CPU (GPU weight 0), the activation and its group sums."""
    ops.host_post2(ids, weights, layer, gpu_weights, x, mailbox, state)


def host_wait2(out, result: int, state):
    """Host tier v2: `out` + the host result of the last post (a copy of `out` when nothing was posted)."""
    return ops.host_wait2(out, result, state)


def host_tier2_start(banks, hidden: int, intermediate: int, group_size: int, cpus, num_experts: int):
    """Host tier v2 CPU workers over the host banks (6 pointers per layer) -> `(handle, mailbox, result)`."""
    return ops.host_tier2_start(list(banks), hidden, intermediate, group_size, list(cpus), num_experts)


def host_tier2_stop(handle: int):
    ops.host_tier2_stop(handle)


def host_tier2_timing(handle: int):
    """`[gate_ns, act_ns, down_ns, calls]` accumulated since the last call (and reset)."""
    return ops.host_tier2_timing(handle)


@torch.library.register_fake(add_op_namespace_prefix("host_wait2"))
def _(out, result, state):
    return torch.empty_like(out)


def host_hint2(ids, layer: int, num_experts: int, table, mailbox: int, hstate):
    """Host tier v2 prefetch hint: the non-resident experts among `ids` (guesses for `layer`) get staged by the CPU
    workers (each its own row slices, into its L2) before the layer posts its real misses."""
    ops.host_hint2(ids, layer, num_experts, table, mailbox, hstate)


def host_fetch2(dst, numel: int, result: int, state):
    """Host tier v2: `dst[:numel]` = the host result of the last post (zeros when nothing was posted)."""
    ops.host_fetch2(dst, numel, result, state)


def host_guess2(x, router_weight, k: int, layer: int, table, mailbox: int, hstate, scratch, counter, max_out: int = 64, blocks: int = 8):
    """Host tier v2: top-`k` of `router_weight @ x` (one token), minus the experts resident for `layer`, posted as the
    prefetch hint for `layer`, in one launch. `scratch` ([E] float) and `counter` (one int32, zero) are static buffers."""
    ops.host_guess2(x, router_weight, k, layer, table, mailbox, hstate, scratch, counter, max_out, blocks)


def host_refresh2(ids, layer: int, num_experts: int, table, owner, age, clock, max_installs: int, mailbox: int, rstate):
    """Host tier v2 pool refresh: commit the installs the CPU installer has copied, plan installs (LRU victims) for this
    layer's missed experts. `rstate`: int64 [produced, committed]."""
    ops.host_refresh2(ids, layer, num_experts, table, owner, age, clock, max_installs, mailbox, rstate)


def host_tier2_installer(handle: int, pool, expert_bytes, num_experts: int, device: int, cpu: int):
    """Start the CPU installer thread: planned installs are copied host bank -> pool slot by the copy engine."""
    ops.host_tier2_installer(handle, list(pool), list(expert_bytes), num_experts, device, cpu)


def host_tier2_drain(handle: int) -> int:
    """Block until every planned install is copied; returns the number of installs so far."""
    return ops.host_tier2_drain(handle)


def gtimer(buf, slot):
    """Diagnostics: write the GPU global timer (ns) into `buf[slot % len(buf)]` and advance `slot` (int64 tensors)."""
    ops.gtimer(buf, slot)


def host_lookup_post3(ids, weights, layer: int, num_experts: int, table, age, clock, x, mailbox: int, state):
    """Host tier v3, one launch on the main stream: lookup (slots / GPU weights, LRU ages) and post the non-resident
    routes to the CPU workers -> `(slots, gpu_weights)`. `state` (int64 [2]) is updated in place."""
    return ops.host_lookup_post3(ids, weights, layer, num_experts, table, age, clock, x, mailbox, state)


def host_join3(out, result: int, state):
    """Host tier v3: `out` + the CPU result of this layer's post (waits on the mapped done flag)."""
    return ops.host_join3(out, result, state)


@torch.library.register_fake(add_op_namespace_prefix("host_lookup_post3"))
def _(ids, weights, layer, num_experts, table, age, clock, x, mailbox, state):
    return torch.empty_like(ids), torch.empty_like(weights)


@torch.library.register_fake(add_op_namespace_prefix("host_join3"))
def _(out, result, state):
    return torch.empty_like(out)


def hc_mix_guess(streams, norm_weight, down, up, eps: float, xhat, scratch):
    """Approximate MoE input of a hyper-connection layer from the streams entering it (no attention): grouped RMSNorm,
    low-rank mixing gates, weighted mean of the streams -> `xhat` [H] (bf16). `scratch`: float [2 * G * H + R]."""
    ops.hc_mix_guess(streams, norm_weight, down, up, eps, xhat, scratch)
