"""Fused sigmoid top-k MoE router for one decode token on Metal: logits = x @ W.T (fp32 weights), scores =
sigmoid(logits), top-k of (scores + correction bias), weights = scores[top-k] / sum * route_scale, in one dispatch (the
threadgroup finishing last does the selection). Replaces mm + sigmoid + topk + gather/normalize (4 dispatches).

    router_topk(x [1, H], weight fp32 [E, H], bias fp32 [E], top_k, route_scale) -> (logits [1, E] fp32,
                                                                                    weights [1, k] fp32,
                                                                                    indices [1, k] int64)
"""

import math

import torch


_SOURCE = """
#include <metal_stdlib>
using namespace metal;

constant constexpr int MAX_E = 256;  // experts the selection holds (8 per lane)
// SPLIT simdgroups share one expert row (strided float4 loads, partial sums reduced in threadgroup memory);
// a 256-thread threadgroup holds 8 / SPLIT experts.
template <typename T, int SPLIT>
kernel void router_topk(device const T* x, device const float* w, device const float* bias, device float* logits,
                        device float* out_w, device long* out_i, device atomic_uint* counter,
                        constant int& E, constant int& H, constant int& K, constant float& scale,
                        uint tg [[threadgroup_position_in_grid]], uint n_tg [[threadgroups_per_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    constexpr int ROWS = 8 / SPLIT;
    threadgroup float partial[8];
    int local = sg / SPLIT, part = sg % SPLIT;
    int e = tg * ROWS + local;
    float acc = 0.0f;
    if (e < E) {
        device const float4* wr = (device const float4*)(w + (long)e * H);
        device const vec<T, 4>* xr = (device const vec<T, 4>*)x;
        for (int i = part * 32 + lane; i < H / 4; i += 32 * SPLIT) acc += dot(wr[i], float4(xr[i]));
        acc = simd_sum(acc);
    }
    if (lane == 0) partial[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0 && lane == 0 && e < E) {
        float total = 0.0f;
        for (int p = 0; p < SPLIT; ++p) total += partial[local * SPLIT + p];
        logits[e] = total;
    }
    // the last threadgroup to finish selects the top-k over all logits
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup bool is_last;
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        is_last = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed) == n_tg - 1;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (!is_last || sg != 0) return;
    if (lane == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);  // ready for the next call

    // lane l holds experts l, l + 32, ... : score (sigmoid) and selection key (score + bias)
    constexpr int PER = MAX_E / 32;
    float score[PER], key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        int idx = lane + 32 * j;
        float s = idx < E ? 1.0f / (1.0f + exp(-logits[idx])) : 0.0f;
        score[j] = s;
        key[j] = idx < E ? s + bias[idx] : -INFINITY;
    }
    float total = 0.0f;
    float picked_score = 0.0f;
    long picked_idx = 0;
    // compile-time indices only (unrolled), so score / key stay in registers
    for (int k = 0; k < K; ++k) {
        float best = -INFINITY, best_s = 0.0f;
        int best_j = 0;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            bool better = key[j] > best;
            best = better ? key[j] : best;
            best_s = better ? score[j] : best_s;
            best_j = better ? j : best_j;
        }
        float top = simd_max(best);
        // lowest expert index among the lanes holding the max
        int candidate = best == top ? int(lane) + 32 * best_j : 0x7fffffff;
        int winner = simd_min(candidate);
        float s = simd_shuffle(best_s, ushort(winner % 32));
        bool mine = int(lane) == winner % 32;
#pragma unroll
        for (int j = 0; j < PER; ++j) key[j] = (mine && j == winner / 32) ? -INFINITY : key[j];
        total += s;
        if (int(lane) == k) { picked_score = s; picked_idx = winner; }
    }
    if (lane < K) {
        out_w[lane] = picked_score / (total + 1e-20f) * scale;
        out_i[lane] = picked_idx;
    }
}

template [[host_name("router_topk_bfloat_1")]] kernel void router_topk<bfloat, 1>(device const bfloat*, device const float*, device const float*, device float*, device float*, device long*, device atomic_uint*, constant int&, constant int&, constant int&, constant float&, uint, uint, uint, uint, uint);
template [[host_name("router_topk_bfloat_2")]] kernel void router_topk<bfloat, 2>(device const bfloat*, device const float*, device const float*, device float*, device float*, device long*, device atomic_uint*, constant int&, constant int&, constant int&, constant float&, uint, uint, uint, uint, uint);
template [[host_name("router_topk_bfloat_4")]] kernel void router_topk<bfloat, 4>(device const bfloat*, device const float*, device const float*, device float*, device float*, device long*, device atomic_uint*, constant int&, constant int&, constant int&, constant float&, uint, uint, uint, uint, uint);
template [[host_name("router_topk_bfloat_8")]] kernel void router_topk<bfloat, 8>(device const bfloat*, device const float*, device const float*, device float*, device float*, device long*, device atomic_uint*, constant int&, constant int&, constant int&, constant float&, uint, uint, uint, uint, uint);
template [[host_name("router_topk_half_1")]] kernel void router_topk<half, 1>(device const half*, device const float*, device const float*, device float*, device float*, device long*, device atomic_uint*, constant int&, constant int&, constant int&, constant float&, uint, uint, uint, uint, uint);
template [[host_name("router_topk_half_2")]] kernel void router_topk<half, 2>(device const half*, device const float*, device const float*, device float*, device float*, device long*, device atomic_uint*, constant int&, constant int&, constant int&, constant float&, uint, uint, uint, uint, uint);
template [[host_name("router_topk_half_4")]] kernel void router_topk<half, 4>(device const half*, device const float*, device const float*, device float*, device float*, device long*, device atomic_uint*, constant int&, constant int&, constant int&, constant float&, uint, uint, uint, uint, uint);
template [[host_name("router_topk_half_8")]] kernel void router_topk<half, 8>(device const half*, device const float*, device const float*, device float*, device float*, device long*, device atomic_uint*, constant int&, constant int&, constant int&, constant float&, uint, uint, uint, uint, uint);
template [[host_name("router_topk_float_1")]] kernel void router_topk<float, 1>(device const float*, device const float*, device const float*, device float*, device float*, device long*, device atomic_uint*, constant int&, constant int&, constant int&, constant float&, uint, uint, uint, uint, uint);
template [[host_name("router_topk_float_2")]] kernel void router_topk<float, 2>(device const float*, device const float*, device const float*, device float*, device float*, device long*, device atomic_uint*, constant int&, constant int&, constant int&, constant float&, uint, uint, uint, uint, uint);
template [[host_name("router_topk_float_4")]] kernel void router_topk<float, 4>(device const float*, device const float*, device const float*, device float*, device float*, device long*, device atomic_uint*, constant int&, constant int&, constant int&, constant float&, uint, uint, uint, uint, uint);
template [[host_name("router_topk_float_8")]] kernel void router_topk<float, 8>(device const float*, device const float*, device const float*, device float*, device float*, device long*, device atomic_uint*, constant int&, constant int&, constant int&, constant float&, uint, uint, uint, uint, uint);
"""

SPLIT = 8  # simdgroups per expert row (sweep: router_v2/bench.py)
_NAMES = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}
_LIB, _COUNTERS = None, {}


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_SOURCE)
    return _LIB


def _torch_router(x, weight, bias, top_k, route_scale):
    logits = torch.nn.functional.linear(x.float().reshape(-1, weight.shape[1]), weight)
    scores = torch.sigmoid(logits)
    indices = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    return logits, weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * route_scale, indices


def _router_mps(x, weight, bias, top_k, route_scale):
    experts, hidden = weight.shape
    if x.numel() != hidden or experts > 256 or top_k > 32 or hidden % 4 or weight.dtype != torch.float32:
        return _torch_router(x, weight, bias, top_k, route_scale)
    counter = _COUNTERS.get(x.device)
    if counter is None:
        counter = _COUNTERS[x.device] = torch.zeros(1, dtype=torch.int32, device=x.device)
    logits = torch.empty(1, experts, dtype=torch.float32, device=x.device)
    weights = torch.empty(1, top_k, dtype=torch.float32, device=x.device)
    indices = torch.empty(1, top_k, dtype=torch.int64, device=x.device)
    groups = math.ceil(experts / (8 // SPLIT))
    getattr(_lib(), f"router_topk_{_NAMES[x.dtype]}_{SPLIT}")(
        x.contiguous(), weight.contiguous(), bias.float().contiguous(), logits, weights, indices, counter,
        experts, hidden, top_k, float(route_scale), threads=groups * 256, group_size=256,
    )  # fmt: skip
    return logits, weights, indices


_OPS = torch.library.Library("metal_router_v2", "DEF")
_OPS.define("router_topk(Tensor x, Tensor weight, Tensor bias, int top_k, float route_scale) -> (Tensor, Tensor, Tensor)")
_OPS.impl("router_topk", _router_mps, "MPS")
_OPS.impl("router_topk", _torch_router, "CPU")


@torch.library.register_fake("metal_router_v2::router_topk")
def _(x, weight, bias, top_k, route_scale):
    tokens = x.numel() // weight.shape[1]
    return (
        x.new_empty(tokens, weight.shape[0], dtype=torch.float32),
        x.new_empty(tokens, top_k, dtype=torch.float32),
        x.new_empty(tokens, top_k, dtype=torch.int64),
    )


router_topk = torch.ops.metal_router_v2.router_topk.default


def inkling2_router(hidden_states, module):
    """transformers' `inkling2_router` on MPS: one decode token's sigmoid top-k routing in one dispatch."""
    if not isinstance(module.score_fn, torch.nn.Sigmoid):
        return _torch_router_module(hidden_states, module)
    return router_topk(
        hidden_states, module.weight, module.e_score_correction_bias, module.top_k, float(module.routed_scaling_factor)
    )


def _torch_router_module(hidden_states, module):
    flat = hidden_states.float().reshape(-1, module.hidden_dim)
    logits = torch.nn.functional.linear(flat, module.weight)
    scores = module.score_fn(logits)
    indices = torch.topk(scores + module.e_score_correction_bias, module.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    return logits, weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * module.routed_scaling_factor, indices
