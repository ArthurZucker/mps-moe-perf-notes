// Fused batch-1 decode kernels for the small-op chains of Inkling-2 / Inkling-2-Loop on Apple GPUs.
//
// At decode every chain below touches a few KB to ~2 MB, so its cost under torch.compile is the number of dispatches
// Inductor emits for it (3-5 us each on Metal), not bytes. Each kernel here replaces one chain with one dispatch:
//
//   attn_res           Inkling2BlockAttentionResidual / Inkling2LoopMessageAttentionResidual, optionally followed by
//                      the weighted RMSNorm(s) that consume its output (input / q,k,v / post-attention layernorms).
//   message_gate_*     Inkling2LoopMessageGate, optionally followed by the weighted RMSNorm that consumes it.
//   short_conv_update  Inkling2ShortConvolution decode step (conv-state update, depthwise conv, `(conv + x) /
//                      sqrt(k + 1)`) plus the decoder layer's residual add.
//   attention_inputs   The per-head tails of the attention input projections: scaled L2 clip of the queries and of
//                      the relative-position states (with the log scale on full layers), short convolution (static
//                      or dynamic) + L2 clip of the keys, short convolution + unweighted RMSNorm of the values.
//
// All arithmetic is fp32; values are rounded to the activation dtype exactly where the reference modules round them.

#include <metal_stdlib>
using namespace metal;

// Sums NV per-thread values over the threadgroup; every thread gets the totals back in `v`.
// `scratch` holds 32 * NV + NV floats: one row of partials per simdgroup, then the totals.
template <uint NV>
inline void threadgroup_sum(thread float* v, threadgroup float* scratch, uint nsg, uint sg, uint lane, uint tid) {
    for (uint i = 0; i < NV; ++i) {
        const float s = simd_sum(v[i]);
        if (lane == 0) {
            scratch[sg * NV + i] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* totals = scratch + 32 * NV;
    if (tid < NV) {
        float s = 0.0f;
        for (uint g = 0; g < nsg; ++g) {
            s += scratch[g * NV + tid];
        }
        totals[tid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < NV; ++i) {
        v[i] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// =====================================================================================================================
// attn_res: attention of n_queries learned fp32 queries over the RMS-normalized stack of NS states
// [block states..., partial].
//
//   x_j = stack[j] * rsqrt(mean(stack[j]^2) + eps)          (j < NS)
//   p_ij = softmax_j(<query_i, x_j> / sqrt(hidden))
//   out_i = sum_j p_ij x_j                                     -> fp32, or with NORM:
//   out_i = TO(weight_i * (out_i * rsqrt(mean(out_i^2) + norm_eps)))
//
// One threadgroup per (query, token), AR_PER_THREAD hidden elements per thread; the threadgroups of the queries run
// in parallel, each recomputing the state norms. On Apple GPUs every dependent round trip to memory costs ~1-2 us,
// far more than this kernel's ~30 KB of traffic, so it is written as exactly one load wave and one store wave: NS is a
// template parameter (every loop unrolls, the whole stack sits in registers), every load is issued up front, and
// <query, x_j> = rsqrt_j * <query, stack[j]> lets one threadgroup reduction (sums of squares and dot products) feed
// everything else. The softmax is recomputed by every thread rather than broadcast.
// =====================================================================================================================

constant constexpr uint AR_PER_THREAD = 3;  // hidden <= 3 * 1024
constant constexpr uint AR_MAX_STATES = 8;

// Sums the NV values of every thread over the threadgroup; every thread gets the totals back in `v`.
// `partials` holds 32 * NV floats, `totals` NV floats.
template <uint NV>
inline void threadgroup_sum_all(thread float (&v)[NV], threadgroup float* partials, threadgroup float* totals,
                                uint nsg, uint sg, uint lane, uint tid) {
    for (uint i = 0; i < NV; ++i) {
        const float s = simd_sum(v[i]);
        if (lane == 0) {
            partials[i * 32 + sg] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = sg; i < NV; i += nsg) {  // one simdgroup per value: lane g holds simdgroup g's partial
        const float s = simd_sum(lane < nsg ? partials[i * 32 + lane] : 0.0f);
        if (lane == 0) {
            totals[i] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < NV; ++i) {
        v[i] = totals[i];
    }
}

template <typename TB, typename TP, typename TO, bool NORM, uint NS>
kernel void attn_res(
    device const TB* blocks,          // [rows, NS - 1, hidden]
    device const TP* partial,         // [rows, hidden]
    device const float* query,        // [n_queries, hidden]
    device const TO* weight,          // NORM: [n_queries or 1, hidden] (row stride `weight_stride`), else unused
    device TO* out,                   // [rows, n_queries, hidden]
    constant int& hidden,
    constant float& eps,
    constant float& norm_eps,
    constant int& weight_stride,
    uint2 tgp [[threadgroup_position_in_grid]],     // x: query, y: token row
    uint2 n_tg [[threadgroups_per_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tg_size2 [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint NV = 2 * NS;
    const uint tid = tid2.x, tg_size = tg_size2.x;
    threadgroup float partials[32 * NV];
    threadgroup float totals[NV];
    const uint nsg = tg_size / 32;
    const uint H = uint(hidden);
    const uint qi = tgp.x, row = tgp.y, nq = n_tg.x;
    device const TB* block_row = blocks + size_t(row) * (NS - 1) * H;
    device const TP* partial_row = partial + size_t(row) * H;
    device const float* q_row = query + qi * H;
    device const TO* w_row = weight + qi * uint(weight_stride);

    // the one load wave
    float x[NS][AR_PER_THREAD], q[AR_PER_THREAD], w[AR_PER_THREAD];
    for (uint e = 0; e < AR_PER_THREAD; ++e) {
        const uint h = tid + e * tg_size;
        const bool valid = h < H;
        for (uint j = 0; j + 1 < NS; ++j) {
            x[j][e] = valid ? float(block_row[j * H + h]) : 0.0f;
        }
        x[NS - 1][e] = valid ? float(partial_row[h]) : 0.0f;
        q[e] = valid ? q_row[h] : 0.0f;
        w[e] = (NORM && valid) ? float(w_row[h]) : 0.0f;
    }

    // sum of squares and query dot product of every state
    float red[NV];  // [2j] sum of squares of state j, [2j + 1] its dot product with the query
    for (uint j = 0; j < NS; ++j) {
        red[2 * j] = red[2 * j + 1] = 0.0f;
        for (uint e = 0; e < AR_PER_THREAD; ++e) {
            red[2 * j] += x[j][e] * x[j][e];
            red[2 * j + 1] += q[e] * x[j][e];
        }
    }
    threadgroup_sum_all<NV>(red, partials, totals, nsg, sg, lane, tid);

    // softmax over the states, folded with the state rsqrt: out = sum_j p[j] * stack[j]
    const float scale = rsqrt(float(H));
    float p[NS];
    float mx = -INFINITY;
    for (uint j = 0; j < NS; ++j) {
        const float rs = rsqrt(red[2 * j] / float(H) + eps);
        red[2 * j] = rs;
        p[j] = red[2 * j + 1] * rs * scale;
        mx = max(mx, p[j]);
    }
    float denom = 0.0f;
    for (uint j = 0; j < NS; ++j) {
        p[j] = precise::exp(p[j] - mx);
        denom += p[j];
    }
    for (uint j = 0; j < NS; ++j) {
        p[j] *= red[2 * j] / denom;
    }
    float o[AR_PER_THREAD];
    float oss[1] = {0.0f};
    for (uint e = 0; e < AR_PER_THREAD; ++e) {
        float acc = 0.0f;
        for (uint j = 0; j < NS; ++j) {
            acc += p[j] * x[j][e];
        }
        o[e] = acc;
        oss[0] += acc * acc;
    }

    float r = 1.0f;
    if (NORM) {
        threadgroup_sum_all<1>(oss, partials, totals, nsg, sg, lane, tid);
        r = rsqrt(oss[0] / float(H) + norm_eps);
    }
    device TO* out_row = out + (size_t(row) * nq + qi) * H;
    for (uint e = 0; e < AR_PER_THREAD; ++e) {
        const uint h = tid + e * tg_size;
        if (h < H) {
            out_row[h] = NORM ? TO(w[e] * (o[e] * r)) : TO(o[e]);
        }
    }
}

#define INSTANTIATE_ATTN_RES(TB, TP, TO, NORM, NS)                                                                     \
    template [[host_name("attn_res_" #TB "_" #TP "_" #NORM "_" #NS)]] kernel void attn_res<TB, TP, TO, NORM, NS>(      \
        device const TB*, device const TP*, device const float*, device const TO*, device TO*, constant int&,          \
        constant float&, constant float&, constant int&, uint2, uint2, uint2, uint2, uint, uint);

#define INSTANTIATE_ATTN_RES_NS(TB, TP, NS)                                                                            \
    INSTANTIATE_ATTN_RES(TB, TP, float, 0, NS)                                                                         \
    INSTANTIATE_ATTN_RES(TB, TP, TB, 1, NS)

#define INSTANTIATE_ATTN_RES_PAIR(TB, TP)                                                                              \
    INSTANTIATE_ATTN_RES_NS(TB, TP, 1)                                                                                 \
    INSTANTIATE_ATTN_RES_NS(TB, TP, 2)                                                                                 \
    INSTANTIATE_ATTN_RES_NS(TB, TP, 3)                                                                                 \
    INSTANTIATE_ATTN_RES_NS(TB, TP, 4)                                                                                 \
    INSTANTIATE_ATTN_RES_NS(TB, TP, 5)                                                                                 \
    INSTANTIATE_ATTN_RES_NS(TB, TP, 6)                                                                                 \
    INSTANTIATE_ATTN_RES_NS(TB, TP, 7)                                                                                 \
    INSTANTIATE_ATTN_RES_NS(TB, TP, 8)

INSTANTIATE_ATTN_RES_PAIR(float, float)
INSTANTIATE_ATTN_RES_PAIR(half, half)
INSTANTIATE_ATTN_RES_PAIR(half, float)
INSTANTIATE_ATTN_RES_PAIR(bfloat, bfloat)
INSTANTIATE_ATTN_RES_PAIR(bfloat, float)

// =====================================================================================================================
// message_gate (one token): Inkling2LoopMessageGate in one dispatch.
//
//   u = TD(cat(h * rsqrt(mean(h^2) + eps), m * rsqrt(mean(m^2) + eps)))      [2 * H]
//   z = float(TD(down @ u))                                                  [rank], down: TD [rank, 2 * H]
//   out = (1 - tanh(forget @ z)) * h + tanh(message @ z) * m                 forget / message: fp32 [H, rank]
//   NORM: out = TD(norm_weight * (out * rsqrt(mean(out^2) + norm_eps)))
//
// MG_TGS threadgroups. A dependent round trip to memory costs ~1-2 us on Apple GPUs, more than streaming this
// kernel's 1.8 MB, so the hidden size is a template parameter: every loop has a constant trip count and unrolls, and
// everything a threadgroup reads (its slices of down / forget / message, h and m) is loaded into registers in one
// wave at the start. What is left are the cross-threadgroup exchanges:
//   1. RMS of h and m (every threadgroup), then one partial z dot product per simdgroup ((z row, column slice) units
//      over the whole grid), published to `exchange`;
//   2. grid sync; every threadgroup sums the partials into z and computes the gates of its H / MG_TGS rows;
//   3. NORM: the threadgroups publish their sums of squares, grid sync, every threadgroup normalizes its own rows.
// The grid sync uses one flag word per threadgroup (an epoch, incremented at every sync) instead of a shared counter:
// each threadgroup stores its own flag and one simdgroup polls all MG_TGS flags with one load per lane, so arrivals
// never contend on an atomic read-modify-write (a shared ticket counter measured ~0.75 us per threadgroup). It relies
// on the grid being co-resident: on an M3 Max, 16 or 18 threadgroups of 1024 threads (and 32 of 256) are, 24 or 32 of
// 512 are not. The spin is bounded, so a grid that is not co-resident gives wrong results instead of hanging the GPU. The grid
// has a fixed size, so the flags stay equal between calls (calls are serialized on the stream; one flag buffer
// serves every call).
// =====================================================================================================================

constant constexpr uint MG_TGS = 16;
constant constexpr uint MG_THREADS = 1024;
constant constexpr uint MG_SGS = MG_THREADS / 32;
constant constexpr uint MG_RANK = 64;
constant constexpr uint MG_SPLIT = MG_TGS * MG_SGS / MG_RANK;  // column slices per z row
constant constexpr uint MG_MAX_SPIN = 1u << 20;

inline void grid_sync(device atomic_uint* flags, uint tg, uint sg, uint lane) {
    threadgroup_barrier(mem_flags::mem_device);
    if (sg == 0) {
        uint epoch = 0;
        if (lane == 0) {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            epoch = atomic_load_explicit(flags + tg, memory_order_relaxed) + 1u;
            atomic_store_explicit(flags + tg, epoch, memory_order_relaxed);
        }
        epoch = simd_broadcast(epoch, 0);
        for (uint spin = 0; spin < MG_MAX_SPIN; ++spin) {  // lane l watches threadgroup l
            const uint seen = lane < MG_TGS ? atomic_load_explicit(flags + lane, memory_order_relaxed) : epoch;
            if (simd_all(int(seen - epoch) >= 0)) {
                break;
            }
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_device);
}

template <typename TD, typename TO, bool NORM, uint H>
kernel void message_gate(
    device const float* hidden_states,        // [H]
    device const float* message,              // [H]
    device const TD* down,                    // [rank, 2 * H]
    device const float* forget,               // [H, rank]
    device const float* message_w,            // [H, rank]
    device const TD* norm_weight,             // NORM: [H], else unused
    device TO* out,                           // [H]
    device atomic_uint* flags,                // [MG_TGS] grid sync epochs
    coherent(device) device float* exchange,  // [rank * MG_SPLIT] partial z, then [MG_TGS] sums of squares
    constant float& eps,
    constant float& norm_eps,
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS = H / MG_TGS;                     // output rows per threadgroup
    constexpr uint SLICE = 2 * H / MG_SPLIT;              // columns of a z row per simdgroup
    constexpr uint DOWN_STEPS = SLICE / 64;               // column pairs per lane
    constexpr uint GATE_STEPS = 2 * ROWS / MG_SGS;        // row dot products per simdgroup
    constexpr uint RMS_STEPS = (H + MG_THREADS - 1) / MG_THREADS;
    static_assert(H % MG_TGS == 0 && SLICE % 64 == 0 && (2 * ROWS) % MG_SGS == 0 && ROWS <= MG_THREADS,
                  "unsupported hidden size");
    threadgroup float scratch[32 * 2 + 2];
    threadgroup float gates[2 * ROWS];
    const uint row0 = tg * ROWS;
    const uint unit = tg * MG_SGS + sg;  // = z row * MG_SPLIT + column slice
    const uint begin = (unit % MG_SPLIT) * SLICE;
    device const TD* down_row = down + size_t(unit / MG_SPLIT) * 2 * H;

    // ---- the load wave
    float hm[2][RMS_STEPS];
#pragma unroll
    for (uint i = 0; i < RMS_STEPS; ++i) {
        const uint k = tid + i * MG_THREADS;
        hm[0][i] = k < H ? hidden_states[k] : 0.0f;
        hm[1][i] = k < H ? message[k] : 0.0f;
    }
    float2 u[DOWN_STEPS];
    vec<TD, 2> wd[DOWN_STEPS];
#pragma unroll
    for (uint t = 0; t < DOWN_STEPS; ++t) {
        const uint k = begin + 2 * lane + 64 * t;  // a pair never straddles the h / m boundary (H is even)
        u[t] = k < H ? *(device const float2*)(hidden_states + k) : *(device const float2*)(message + k - H);
        wd[t] = *(device const vec<TD, 2>*)(down_row + k);
    }
    float2 wg[GATE_STEPS];  // row dot product d = sg + MG_SGS * t: row d / 2 of forget (d even) or message (d odd)
#pragma unroll
    for (uint t = 0; t < GATE_STEPS; ++t) {
        const uint d = sg + MG_SGS * t;
        wg[t] = *(device const float2*)((d % 2 ? message_w : forget) + size_t(row0 + d / 2) * MG_RANK + 2 * lane);
    }
    float h_row = 0.0f, m_row = 0.0f, w_row = 0.0f;
    if (tid < ROWS) {
        h_row = hidden_states[row0 + tid];
        m_row = message[row0 + tid];
        w_row = NORM ? float(norm_weight[row0 + tid]) : 0.0f;
    }

    // ---- 1. RMS of h and m, partial z dot product of this simdgroup's unit
    float ss[2] = {0.0f, 0.0f};
#pragma unroll
    for (uint i = 0; i < RMS_STEPS; ++i) {
        ss[0] += hm[0][i] * hm[0][i];
        ss[1] += hm[1][i] * hm[1][i];
    }
    threadgroup_sum<2>(ss, scratch, MG_SGS, sg, lane, tid);
    const float rh = rsqrt(ss[0] / float(H) + eps), rm = rsqrt(ss[1] / float(H) + eps);
    float acc = 0.0f;
#pragma unroll
    for (uint t = 0; t < DOWN_STEPS; ++t) {
        const float r = begin + 2 * lane + 64 * t < H ? rh : rm;
        acc += dot(float2(vec<TD, 2>(u[t] * r)), float2(wd[t]));
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        exchange[unit] = acc;
    }
    grid_sync(flags, tg, sg, lane);

    // ---- 2. z (lane l: z[2l], z[2l + 1], rounded as nn.Linear returns it), then the gates of this threadgroup's rows
    float2 z = 0.0f;
#pragma unroll
    for (uint s = 0; s < MG_SPLIT; ++s) {
        z += float2(exchange[(2 * lane) * MG_SPLIT + s], exchange[(2 * lane + 1) * MG_SPLIT + s]);
    }
    z = float2(vec<TD, 2>(z));
#pragma unroll
    for (uint t = 0; t < GATE_STEPS; ++t) {
        const float g = simd_sum(dot(wg[t], z));
        if (lane == 0) {
            gates[sg + MG_SGS * t] = g;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = 0.0f;
    if (tid < ROWS) {
        o = (1.0f - precise::tanh(gates[2 * tid])) * h_row + precise::tanh(gates[2 * tid + 1]) * m_row;
        if (!NORM) {
            out[row0 + tid] = TO(o);
        }
    }
    if (!NORM) {
        return;
    }

    // ---- 3. NORM: grid-wide sum of squares, then this threadgroup's rows
    float oss[1] = {o * o};
    threadgroup_sum<1>(oss, scratch, MG_SGS, sg, lane, tid);
    if (tid == 0) {
        exchange[MG_RANK * MG_SPLIT + tg] = oss[0];
    }
    grid_sync(flags, tg, sg, lane);
    threadgroup float total;
    if (sg == 0) {
        const float s = simd_sum(lane < MG_TGS ? exchange[MG_RANK * MG_SPLIT + lane] : 0.0f);
        if (lane == 0) {
            total = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < ROWS) {
        out[row0 + tid] = TO(w_row * (o * rsqrt(total / float(H) + norm_eps)));
    }
}

template <typename TD, typename TO, bool NORM, uint H>
kernel void message_gate3(
    device const float* h0, device const float* h1, device const float* h2,
    device const float* m0, device const float* m1, device const float* m2,
    device const TD* d0, device const TD* d1, device const TD* d2,
    device const float* f0, device const float* f1, device const float* f2,
    device const float* w0, device const float* w1, device const float* w2,
    device const TD* n0, device const TD* n1, device const TD* n2,  // NORM: the weighted RMSNorm after each gate
    device TO* o0, device TO* o1, device TO* o2,
    device atomic_uint* flags_all,                // [3 * MG_TGS]
    coherent(device) device float* exchange_all,  // [3 * (rank * MG_SPLIT + MG_TGS)]
    constant float& eps,
    constant float& norm_eps,
    uint tg_all [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    // three independent message gates (query / key / value), MG_TGS threadgroups each,
    // each gate syncing only its own threadgroups (own flags and exchange slice)
    const uint g = tg_all / MG_TGS, tg = tg_all % MG_TGS;
    device const float* hidden_states = g == 0 ? h0 : (g == 1 ? h1 : h2);
    device const TD* norm_weight = g == 0 ? n0 : (g == 1 ? n1 : n2);
    device const float* message = g == 0 ? m0 : (g == 1 ? m1 : m2);
    device const TD* down = g == 0 ? d0 : (g == 1 ? d1 : d2);
    device const float* forget = g == 0 ? f0 : (g == 1 ? f1 : f2);
    device const float* message_w = g == 0 ? w0 : (g == 1 ? w1 : w2);
    device TO* out = g == 0 ? o0 : (g == 1 ? o1 : o2);
    device atomic_uint* flags = flags_all + g * MG_TGS;
    coherent(device) device float* exchange = exchange_all + g * (MG_RANK * MG_SPLIT + MG_TGS);

    constexpr uint ROWS = H / MG_TGS;                     // output rows per threadgroup
    constexpr uint SLICE = 2 * H / MG_SPLIT;              // columns of a z row per simdgroup
    constexpr uint DOWN_STEPS = SLICE / 64;               // column pairs per lane
    constexpr uint GATE_STEPS = 2 * ROWS / MG_SGS;        // row dot products per simdgroup
    constexpr uint RMS_STEPS = (H + MG_THREADS - 1) / MG_THREADS;
    static_assert(H % MG_TGS == 0 && SLICE % 64 == 0 && (2 * ROWS) % MG_SGS == 0 && ROWS <= MG_THREADS,
                  "unsupported hidden size");
    threadgroup float scratch[32 * 2 + 2];
    threadgroup float gates[2 * ROWS];
    const uint row0 = tg * ROWS;
    const uint unit = tg * MG_SGS + sg;  // = z row * MG_SPLIT + column slice
    const uint begin = (unit % MG_SPLIT) * SLICE;
    device const TD* down_row = down + size_t(unit / MG_SPLIT) * 2 * H;

    // ---- the load wave
    float hm[2][RMS_STEPS];
#pragma unroll
    for (uint i = 0; i < RMS_STEPS; ++i) {
        const uint k = tid + i * MG_THREADS;
        hm[0][i] = k < H ? hidden_states[k] : 0.0f;
        hm[1][i] = k < H ? message[k] : 0.0f;
    }
    float2 u[DOWN_STEPS];
    vec<TD, 2> wd[DOWN_STEPS];
#pragma unroll
    for (uint t = 0; t < DOWN_STEPS; ++t) {
        const uint k = begin + 2 * lane + 64 * t;  // a pair never straddles the h / m boundary (H is even)
        u[t] = k < H ? *(device const float2*)(hidden_states + k) : *(device const float2*)(message + k - H);
        wd[t] = *(device const vec<TD, 2>*)(down_row + k);
    }
    float2 wg[GATE_STEPS];  // row dot product d = sg + MG_SGS * t: row d / 2 of forget (d even) or message (d odd)
#pragma unroll
    for (uint t = 0; t < GATE_STEPS; ++t) {
        const uint d = sg + MG_SGS * t;
        wg[t] = *(device const float2*)((d % 2 ? message_w : forget) + size_t(row0 + d / 2) * MG_RANK + 2 * lane);
    }
    float h_row = 0.0f, m_row = 0.0f, w_row = 0.0f;
    if (tid < ROWS) {
        h_row = hidden_states[row0 + tid];
        m_row = message[row0 + tid];
        w_row = NORM ? float(norm_weight[row0 + tid]) : 0.0f;
    }

    // ---- 1. RMS of h and m, partial z dot product of this simdgroup's unit
    float ss[2] = {0.0f, 0.0f};
#pragma unroll
    for (uint i = 0; i < RMS_STEPS; ++i) {
        ss[0] += hm[0][i] * hm[0][i];
        ss[1] += hm[1][i] * hm[1][i];
    }
    threadgroup_sum<2>(ss, scratch, MG_SGS, sg, lane, tid);
    const float rh = rsqrt(ss[0] / float(H) + eps), rm = rsqrt(ss[1] / float(H) + eps);
    float acc = 0.0f;
#pragma unroll
    for (uint t = 0; t < DOWN_STEPS; ++t) {
        const float r = begin + 2 * lane + 64 * t < H ? rh : rm;
        acc += dot(float2(vec<TD, 2>(u[t] * r)), float2(wd[t]));
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        exchange[unit] = acc;
    }
    grid_sync(flags, tg, sg, lane);

    // ---- 2. z (lane l: z[2l], z[2l + 1], rounded as nn.Linear returns it), then the gates of this threadgroup's rows
    float2 z = 0.0f;
#pragma unroll
    for (uint s = 0; s < MG_SPLIT; ++s) {
        z += float2(exchange[(2 * lane) * MG_SPLIT + s], exchange[(2 * lane + 1) * MG_SPLIT + s]);
    }
    z = float2(vec<TD, 2>(z));
#pragma unroll
    for (uint t = 0; t < GATE_STEPS; ++t) {
        const float g = simd_sum(dot(wg[t], z));
        if (lane == 0) {
            gates[sg + MG_SGS * t] = g;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = 0.0f;
    if (tid < ROWS) {
        o = (1.0f - precise::tanh(gates[2 * tid])) * h_row + precise::tanh(gates[2 * tid + 1]) * m_row;
        if (!NORM) {
            out[row0 + tid] = TO(o);
        }
    }
    if (!NORM) {
        return;
    }

    // ---- 3. NORM: grid-wide sum of squares, then this threadgroup's rows
    float oss[1] = {o * o};
    threadgroup_sum<1>(oss, scratch, MG_SGS, sg, lane, tid);
    if (tid == 0) {
        exchange[MG_RANK * MG_SPLIT + tg] = oss[0];
    }
    grid_sync(flags, tg, sg, lane);
    threadgroup float total;
    if (sg == 0) {
        const float s = simd_sum(lane < MG_TGS ? exchange[MG_RANK * MG_SPLIT + lane] : 0.0f);
        if (lane == 0) {
            total = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < ROWS) {
        out[row0 + tid] = TO(w_row * (o * rsqrt(total / float(H) + norm_eps)));
    }
}

#define INSTANTIATE_MESSAGE_GATE3(TD, H)                                                                               \
    template [[host_name("message_gate3_" #TD "_0_" #H)]] kernel void message_gate3<TD, float, false, H>(              \
        device const float*, device const float*, device const float*, device const float*, device const float*, device const float*, device const TD*,         \
        device const TD*, device const TD*, device const float*, device const float*, device const float*,             \
        device const float*, device const float*, device const float*, device const TD*, device const TD*, device const TD*, device float*, \
        device float*, device float*, device atomic_uint*, coherent(device) device float*, constant float&, constant float&, uint,    \
        uint, uint, uint);                                                                                             \
    template [[host_name("message_gate3_" #TD "_1_" #H)]] kernel void message_gate3<TD, TD, true, H>(                  \
        device const float*, device const float*, device const float*, device const float*, device const float*, device const float*, device const TD*,         \
        device const TD*, device const TD*, device const float*, device const float*, device const float*,             \
        device const float*, device const float*, device const float*, device const TD*, device const TD*, device const TD*, device TD*,  \
        device TD*, device TD*, device atomic_uint*, coherent(device) device float*, constant float&, constant float&, uint, uint,  \
        uint, uint);
#define INSTANTIATE_MESSAGE_GATE(TD, H)                                                                                \
    template [[host_name("message_gate_" #TD "_0_" #H)]] kernel void message_gate<TD, float, false, H>(                \
        device const float*, device const float*, device const TD*, device const float*, device const float*,          \
        device const TD*, device float*, device atomic_uint*, coherent(device) device float*, constant float&,         \
        constant float&, uint, uint, uint, uint);                                                                      \
    template [[host_name("message_gate_" #TD "_1_" #H)]] kernel void message_gate<TD, TD, true, H>(                    \
        device const float*, device const float*, device const TD*, device const float*, device const float*,          \
        device const TD*, device TD*, device atomic_uint*, coherent(device) device float*, constant float&,            \
        constant float&, uint, uint, uint, uint);

// Inkling-2-Mini (the loop model) hidden size; other sizes take the torch path
INSTANTIATE_MESSAGE_GATE(float, 2304)
INSTANTIATE_MESSAGE_GATE(half, 2304)
INSTANTIATE_MESSAGE_GATE(bfloat, 2304)

// =====================================================================================================================
// short_conv_update: one decode step of Inkling2ShortConvolution plus the residual add that follows it.
//
//   state <- [state[1:], x]                                   fp32 [rows, channels, kernel_size], in place
//   y = TX((sum_k weight[c, k] * state[k] + x) / denom)        denom = sqrt(kernel_size + 1)
//   out = TO(float(y) + float(residual))                       (or TO(float(y)) without residual)
// =====================================================================================================================

template <typename TX, typename TR, typename TO, bool RES>
kernel void short_conv_update(
    device const TX* x,          // [rows, channels]
    device float* state,         // [rows, channels, kernel_size]
    device const float* weight,  // [channels, kernel_size]
    device const TR* residual,   // [rows, channels] when RES
    device TO* out,              // [rows, channels]
    constant int& channels,
    constant int& kernel_size,
    constant float& denom,
    uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, row = gid.y, C = uint(channels), K = uint(kernel_size);
    if (c >= C) {
        return;
    }
    const size_t i = size_t(row) * C + c;
    device float* s = state + i * K;
    device const float* w = weight + c * K;
    const float xf = float(x[i]);
    float acc = 0.0f;
    for (uint k = 0; k + 1 < K; ++k) {
        const float v = s[k + 1];
        s[k] = v;
        acc += w[k] * v;
    }
    s[K - 1] = xf;
    acc += w[K - 1] * xf;
    float y = float(TX((acc + xf) / denom));
    if (RES) {
        y += float(residual[i]);
    }
    out[i] = TO(y);
}

#define INSTANTIATE_SHORT_CONV(TX, TR, TO, RES)                                                                        \
    template [[host_name("short_conv_update_" #TX "_" #TR "_" #TO "_" #RES)]] kernel void                              \
    short_conv_update<TX, TR, TO, RES>(device const TX*, device float*, device const float*, device const TR*,         \
                                       device TO*, constant int&, constant int&, constant float&, uint2);

// (post-attention: bf16 residual -> fp32 residual stream, post-MLP: fp32 residual -> bf16 layer output)
#define INSTANTIATE_SHORT_CONV_DTYPE(T)                                                                                \
    INSTANTIATE_SHORT_CONV(T, T, float, 0)                                                                             \
    INSTANTIATE_SHORT_CONV(T, T, float, 1)                                                                             \
    INSTANTIATE_SHORT_CONV(T, float, T, 1)                                                                             \
    INSTANTIATE_SHORT_CONV(T, T, T, 0)                                                                                 \
    INSTANTIATE_SHORT_CONV(T, T, T, 1)

INSTANTIATE_SHORT_CONV(float, float, float, 0)
INSTANTIATE_SHORT_CONV(float, float, float, 1)
INSTANTIATE_SHORT_CONV_DTYPE(half)
INSTANTIATE_SHORT_CONV_DTYPE(bfloat)

// =====================================================================================================================
// attention_inputs: the per-head tails of Inkling2QueryProjection, Inkling2KeyValueProjection (keys and values) and
// the relative-position states of Inkling2RelativePositionBias, for one token, in one dispatch. One simdgroup per
// head vector; threadgroup x indexes [query heads | key heads | value heads | relative heads], y the token row.
//
//   queries / relative states:  y = TX(l2clip(x * scale, threshold))          (LOG: y = TX(float(y) * tau))
//   keys:                       c = TX(conv(x)), y = TX(l2clip(c, threshold))
//   values:                     c = TX(conv(x)), y = TX(c * rsqrt(mean(c^2) + eps))
//   l2clip(v, t) = v * min(t / (|v| + 1e-6), 1),  tau = 1 + alpha * log(max((position + 1) / n_floor, 1))
//
// Static convolution (sliding layers, Inkling2ShortConvolution): c = (sum_k w[c, k] state[k] + x) / sqrt(K + 1),
// w fp32 [C, K]. Dynamic convolution (full layers, Inkling2DynamicShortConvolution): c = (sum_lag (w[lag, c] +
// dyn[head, lag]) state[K - 1 - lag] + x) / sqrt(2K + 1), w = static_weight [K, C], dyn = weight_proj output
// [kv_heads * K]. The conv state update is the same in both: state <- [state[1:], x].
// =====================================================================================================================

constant constexpr uint AI_PER_LANE = 4;  // head_dim <= 128
constant constexpr uint AI_MAX_KERNEL = 8;

template <typename TX, typename TW, bool DYN, bool LOG>
kernel void attention_inputs(
    device const TX* q,                  // [rows, q_heads * head_dim]
    device const TX* k,                  // [rows, kv_heads * head_dim]
    device const TX* v,                  // [rows, kv_heads * head_dim]
    device const TX* rel,                // [rows, q_heads * rel_dim]
    device const TX* q_scale,            // [q_heads * head_dim]
    device const TX* rel_scale,          // [q_heads * rel_dim]
    device float* k_state,               // [rows, kv_heads * head_dim, K]
    device float* v_state,
    device const TW* k_weight,           // static: [C, K] (fp32), dynamic: static_weight [K, C]
    device const TW* v_weight,
    device const TX* k_dyn,              // dynamic: [rows, kv_heads * K], else unused
    device const TX* v_dyn,
    device const long* position_ids,     // LOG: [rows], else unused
    device TX* out,                      // [rows, (q_heads + 2 * kv_heads) * head_dim + q_heads * rel_dim]
    constant int& q_heads,
    constant int& kv_heads,
    constant int& head_dim,
    constant int& rel_dim,
    constant int& kernel_size,
    constant float& qk_threshold,
    constant float& rel_threshold,
    constant float& eps,
    constant float& log_alpha,
    constant float& log_n_floor,
    uint2 tgp [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint Hq = uint(q_heads), Hkv = uint(kv_heads), D = uint(head_dim), R = uint(rel_dim), K = uint(kernel_size);
    const uint head = tgp.x, row = tgp.y;
    const uint C = Hkv * D;
    const uint out_stride = (Hq + 2 * Hkv) * D + Hq * R;
    device TX* out_row = out + size_t(row) * out_stride;

    float tau = 1.0f;
    if (LOG) {
        tau = 1.0f + log_alpha * precise::log(max(float(position_ids[row] + 1) / log_n_floor, 1.0f));
    }

    float val[AI_PER_LANE];
    if (head < Hkv * 2 + Hq && head >= Hq) {
        // keys / values: convolution step, then the head-wise norm
        const bool is_key = head < Hq + Hkv;
        const uint kv_head = head - Hq - (is_key ? 0 : Hkv);
        device const TX* x_row = (is_key ? k : v) + size_t(row) * C;
        device float* state = (is_key ? k_state : v_state) + size_t(row) * C * K;
        device const TW* w = is_key ? k_weight : v_weight;
        float dyn[AI_MAX_KERNEL];
        if (DYN) {
            device const TX* d = (is_key ? k_dyn : v_dyn) + size_t(row) * Hkv * K + kv_head * K;
            for (uint lag = 0; lag < AI_MAX_KERNEL; ++lag) {
                dyn[lag] = lag < K ? float(d[lag]) : 0.0f;
            }
        }
        const float denom = DYN ? sqrt(float(2 * K + 1)) : sqrt(float(K + 1));
        float ss = 0.0f;
        for (uint e = 0; e < AI_PER_LANE; ++e) {
            const uint d = lane + 32 * e;
            float c = 0.0f;
            if (d < D) {
                const uint ch = kv_head * D + d;
                device float* s = state + ch * K;
                const float xf = float(x_row[ch]);
                float acc = 0.0f;
                for (uint t = 0; t + 1 < K; ++t) {  // t: oldest -> newest, lag = K - 1 - t
                    const float sv = s[t + 1];
                    s[t] = sv;
                    acc += sv * (DYN ? float(w[(K - 1 - t) * C + ch]) + dyn[K - 1 - t] : float(w[ch * K + t]));
                }
                s[K - 1] = xf;
                acc += xf * (DYN ? float(w[ch]) + dyn[0] : float(w[ch * K + K - 1]));
                c = float(TX((acc + xf) / denom));
            }
            val[e] = c;
            ss += c * c;
        }
        ss = simd_sum(ss);
        const float factor = is_key ? min(qk_threshold / (sqrt(ss) + 1e-6f), 1.0f) : rsqrt(ss / float(D) + eps);
        device TX* o = out_row + (Hq + (is_key ? 0 : Hkv)) * D + kv_head * D;
        for (uint e = 0; e < AI_PER_LANE; ++e) {
            const uint d = lane + 32 * e;
            if (d < D) {
                o[d] = TX(val[e] * factor);
            }
        }
    } else {
        // queries / relative states: learned scale, L2 clip, log scale on full layers
        const bool is_query = head < Hq;
        const uint qh = is_query ? head : head - Hq - 2 * Hkv;
        const uint dim = is_query ? D : R;
        device const TX* x = (is_query ? q + size_t(row) * Hq * D : rel + size_t(row) * Hq * R) + qh * dim;
        device const TX* scale = (is_query ? q_scale : rel_scale) + qh * dim;
        float ss = 0.0f;
        for (uint e = 0; e < AI_PER_LANE; ++e) {
            const uint d = lane + 32 * e;
            val[e] = d < dim ? float(x[d]) * float(scale[d]) : 0.0f;
            ss += val[e] * val[e];
        }
        ss = simd_sum(ss);
        const float factor = min((is_query ? qk_threshold : rel_threshold) / (sqrt(ss) + 1e-6f), 1.0f);
        device TX* o = is_query ? out_row + qh * D : out_row + (Hq + 2 * Hkv) * D + qh * R;
        for (uint e = 0; e < AI_PER_LANE; ++e) {
            const uint d = lane + 32 * e;
            if (d < dim) {
                const float y = float(TX(val[e] * factor));
                o[d] = TX(LOG ? y * tau : y);
            }
        }
    }
}

#define INSTANTIATE_ATTENTION_INPUTS(TX, TW, DYN, LOG)                                                                 \
    template [[host_name("attention_inputs_" #TX "_" #TW "_" #DYN)]] kernel void attention_inputs<TX, TW, DYN, LOG>(   \
        device const TX*, device const TX*, device const TX*, device const TX*, device const TX*, device const TX*,    \
        device float*, device float*, device const TW*, device const TW*, device const TX*, device const TX*,          \
        device const long*, device TX*, constant int&, constant int&, constant int&, constant int&, constant int&,     \
        constant float&, constant float&, constant float&, constant float&, constant float&, uint2, uint);

// sliding layers: static fp32 conv, no log scale; full layers: dynamic conv (static_weight in the model dtype or fp32)
// and log scale
INSTANTIATE_ATTENTION_INPUTS(float, float, 0, false)
INSTANTIATE_ATTENTION_INPUTS(float, float, 1, true)
INSTANTIATE_ATTENTION_INPUTS(half, float, 0, false)
INSTANTIATE_ATTENTION_INPUTS(half, half, 1, true)
INSTANTIATE_ATTENTION_INPUTS(half, float, 1, true)
INSTANTIATE_ATTENTION_INPUTS(bfloat, float, 0, false)
INSTANTIATE_ATTENTION_INPUTS(bfloat, bfloat, 1, true)
INSTANTIATE_ATTENTION_INPUTS(bfloat, float, 1, true)
INSTANTIATE_MESSAGE_GATE3(float, 2304)
INSTANTIATE_MESSAGE_GATE3(half, 2304)
INSTANTIATE_MESSAGE_GATE3(bfloat, 2304)
