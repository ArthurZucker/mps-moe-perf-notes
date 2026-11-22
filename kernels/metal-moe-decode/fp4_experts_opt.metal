// MoE expert kernels on Metal straight from checkpoint-native NVFP4 / MXFP8 banks (no dequantized copy).
//
// Formats (ModelOpt / OCP):
//   FMT 0 = NVFP4: uint8 [E, N, K / 2], two E2M1 codes per byte (even element in the low nibble), one float8 e4m3
//           scale per 16 values [E, N, K / 16], a float32 global scale per expert (gate / up: [E, 2], down: [E]).
//   FMT 1 = MXFP8: float8 e4m3 [E, N, K], one e8m0 scale (uint8, power of two biased by 127) per 32 values.
//
// Inputs are T tokens x R routes = P (token, route) pairs; `ids` / `wt` are [T, R] (pair p = t * R + r), read in the
// router's dtypes (see route_id / route_wt).
//
//   gate_up_act          one simdgroup = one pair x ROWS outputs: gate row i and up row I + i of the pair's expert are
//                        read by the same lanes, so the activation is applied in registers.
//   experts_fused        both in one dispatch (per 64-wide intermediate slice: activations, then their down columns
//                        into an fp32 workspace with atomics); on par at T = 1, slower masked / at T = 8.
//   down_sum             one threadgroup = one token x ROWS output rows; its SG simdgroups stride together over the
//                        token's (active route, chunk) pairs (zero-weight routes compacted away first) and reduce in
//                        threadgroup memory.
//
// Grids are ordered row-block-major: all pairs (tokens) working on the same rows run next to each other, so when
// several tokens share an expert (speculative verify, T = 8..16) its rows come from the cache after the first read.
// That gets the DRAM traffic down to the distinct experts without a gather / grouping step (a kernel reading each
// distinct expert once for all its tokens measured 1.3-2.6x slower: too few simdgroups, serialized per-token dots).
//
// Dequantization: an E2M1 code (s e e m) or an e4m3 byte (s eeee mmm) moved into the matching bits of a half is the
// value times 2^-14 / 2^-8; the factors are folded into the block scales.

#include <metal_stdlib>
using namespace metal;

inline float2 e2m1x2(uint p) { return float2(as_type<half2>(((p & 0x00080008u) << 12) | ((p & 0x00070007u) << 9))); }
inline float2 e4m3x2(uint p) { return float2(as_type<half2>(((p & 0x00800080u) << 8) | ((p & 0x007F007Fu) << 7))); }

// NVFP4 block scale (e4m3 byte) times 2^14 * 2^8 (undoes both half-bit tricks)
inline float nv_scale(uint b) { return e4m3x2(b).x * 4194304.0f; }
// MXFP8 block scale (e8m0 byte) times 2^8 (undoes the e4m3 half-bit trick)
inline float mx_scale(uint b) { return as_type<float>((b + 8u) << 23); }

// 8 E2M1 codes of word q (nibble j = element j), times s, into out[0..7]
inline void fp4x8(uint q, float s, thread float* out) {
    float2 p0 = e2m1x2(q), p1 = e2m1x2(q >> 4), p2 = e2m1x2(q >> 8), p3 = e2m1x2(q >> 12);
    out[0] = p0.x * s; out[1] = p1.x * s; out[2] = p2.x * s; out[3] = p3.x * s;
    out[4] = p0.y * s; out[5] = p1.y * s; out[6] = p2.y * s; out[7] = p3.y * s;
}

// 4 e4m3 bytes of word q (byte j = element j), times s, into out[0..3]
inline void fp8x4(uint q, float s, thread float* out) {
    float2 lo = e4m3x2(q), hi = e4m3x2(q >> 8);  // (b0, b2), (b1, b3)
    out[0] = lo.x * s; out[1] = hi.x * s; out[2] = lo.y * s; out[3] = hi.y * s;
}

// Chunk c (VPL consecutive values) of a row, dequantized with its block scale. `wrow` / `srow` point at the row.
template <int FMT, int VPL>
inline void decode(device const uchar* wrow, device const uchar* srow, int c, thread float* out) {
    if (FMT == 0) {
        if (VPL == 8) {
            fp4x8(((device const uint*)wrow)[c], nv_scale(srow[c >> 1]), out);
        } else if (VPL == 16) {
            uint2 q = ((device const uint2*)wrow)[c];
            float s = nv_scale(srow[c]);
            fp4x8(q.x, s, out);
            fp4x8(q.y, s, out + 8);
        } else {
            uint4 q = ((device const uint4*)wrow)[c];
            ushort sc = ((device const ushort*)srow)[c];
            float s0 = nv_scale(sc & 0xFF), s1 = nv_scale(sc >> 8);
            fp4x8(q.x, s0, out);
            fp4x8(q.y, s0, out + 8);
            fp4x8(q.z, s1, out + 16);
            fp4x8(q.w, s1, out + 24);
        }
    } else {
        float s = mx_scale(srow[c * VPL / 32]);
        if (VPL == 8) {
            uint2 q = ((device const uint2*)wrow)[c];
            fp8x4(q.x, s, out);
            fp8x4(q.y, s, out + 4);
        } else {
            for (int v = 0; v < VPL / 16; ++v) {
                uint4 q = ((device const uint4*)wrow)[c * (VPL / 16) + v];
                fp8x4(q.x, s, out + 16 * v);
                fp8x4(q.y, s, out + 16 * v + 4);
                fp8x4(q.z, s, out + 16 * v + 8);
                fp8x4(q.w, s, out + 16 * v + 12);
            }
        }
    }
}

template <int FMT>
inline long row_bytes(int k) { return FMT == 0 ? k / 2 : k; }
template <int FMT>
inline long scale_bytes(int k) { return FMT == 0 ? k / 16 : k / 32; }

template <int VPL, typename T>
inline void load_x(device const T* src, thread float* out) {
    device const vec<T, 4>* v = (device const vec<T, 4>*)src;
    for (int i = 0; i < VPL / 4; ++i) {
        float4 f = float4(v[i]);
        out[4 * i] = f.x; out[4 * i + 1] = f.y; out[4 * i + 2] = f.z; out[4 * i + 3] = f.w;
    }
}

template <int VPL>
inline float dotv(thread const float* a, thread const float* b) {
    float d = 0.0f;
    for (int i = 0; i < VPL; ++i) d += a[i] * b[i];
    return d;
}

// Row `row` (K inputs) chunk c dotted with x chunk xs
template <int FMT, int VPL>
inline float chunk_dot(device const uchar* w, device const uchar* s, long row, int k, int c, thread const float* xs) {
    float wv[VPL];
    decode<FMT, VPL>(w + row * row_bytes<FMT>(k), s + row * scale_bytes<FMT>(k), c, wv);
    return dotv<VPL>(wv, xs);
}

// Routing tensors are read as the caller has them (no cast dispatches): `ids` int32 or int64 (the low word, ids are
// small and little-endian), `wt` float32, bfloat16 or float16. `mode` = (int64 ids) | (weight kind << 1), kind 0 / 1 /
// 2 = float32 / bfloat16 / float16.
inline int route_id(device const uint* ids, int p, int mode) { return int(ids[(mode & 1) ? 2 * p : p]); }
inline float route_wt(device const uchar* wt, int p, int mode) {
    const int kind = mode >> 1;
    if (kind == 0) return ((device const float*)wt)[p];
    if (kind == 1) return float(((device const bfloat*)wt)[p]);
    return float(((device const half*)wt)[p]);
}

inline float softcap(float v, float cap) { return cap > 0.0f ? cap * precise::tanh(v / cap) : v; }
inline float activation(float g, float u, float gate_cap, float up_cap) {
    return softcap(g, gate_cap) * (1.0f / (1.0f + exp(-g))) * softcap(u, up_cap);
}

// act[p, i] for the pairs with a nonzero weight (zeros for the others, nothing read). Grid: P * (I / ROWS)
// simdgroups, simdgroup s -> row block s / P, pair s % P.
template <int FMT, typename T, int ROWS, int VPL>
kernel void gate_up_act(device const uchar* w, device const uchar* s, device const float* factor, device const T* x,
                        device const uint* ids, device const uchar* wt, device T* act, constant int& I,
                        constant int& K, constant int& P, constant int& R, constant int& mode,
                        constant float& gate_cap, constant float& up_cap, uint gid [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const int sidx = gid / 32, blk = sidx / P, p = sidx % P, i0 = blk * ROWS;
    if (i0 >= I) return;
    if (route_wt(wt, p, mode) == 0.0f) {  // masked pair (streamed-experts sentinel)
        if (lane < ROWS) act[(long)p * I + i0 + lane] = T(0);
        return;
    }
    const int e = route_id(ids, p, mode), chunks = K / VPL;
    device const T* xt = x + (long)(p / R) * K;
    const long gate_row = (long)e * 2 * I + i0, up_row = gate_row + I;
    float acc[2 * ROWS];
    for (int j = 0; j < 2 * ROWS; ++j) acc[j] = 0.0f;
    for (int c = lane; c < chunks; c += 32) {
        float xs[VPL];
        load_x<VPL>(xt + c * VPL, xs);
        for (int j = 0; j < ROWS; ++j) {
            acc[j] += chunk_dot<FMT, VPL>(w, s, gate_row + j, K, c, xs);
            acc[ROWS + j] += chunk_dot<FMT, VPL>(w, s, up_row + j, K, c, xs);
        }
    }
    for (int j = 0; j < 2 * ROWS; ++j) acc[j] = simd_sum(acc[j]);
    if (lane < ROWS) {
        float g = 0.0f, u = 0.0f;
        for (int j = 0; j < ROWS; ++j)
            if (lane == uint(j)) { g = acc[j]; u = acc[ROWS + j]; }
        if (FMT == 0) { g *= factor[2 * e]; u *= factor[2 * e + 1]; }
        act[(long)p * I + i0 + lane] = T(activation(g, u, gate_cap, up_cap));
    }
}

// y[t, h] = sum_r wt[t, r] * factor[e] * (down[e, h] . act[t, r]), e = ids[t, r]. Grid: (H / ROWS) * NT threadgroups
// of SG simdgroups, threadgroup g -> row block g / NT, token g % NT.
template <int FMT, typename T, int ROWS, int VPL, int SG>
kernel void down_sum(device const uchar* w, device const uchar* s, device const float* factor, device const T* act,
                     device const uint* ids, device const uchar* wt, device T* y, constant int& H, constant int& I,
                     constant int& R, constant int& NT, constant int& mode, uint tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int MAX_ROUTES = 64, THREADS = 32 * SG;
    threadgroup int route_e[MAX_ROUTES], route[MAX_ROUTES];
    threadgroup float route_w[MAX_ROUTES];
    threadgroup int active;
    threadgroup float partial[SG][ROWS];
    const int t = tg % NT, h0 = (tg / NT) * ROWS;
    if (tid == 0) {  // compact the token's nonzero-weight routes, global scale folded in
        int n = 0;
        for (int r = 0; r < R; ++r) {
            const float v = route_wt(wt, t * R + r, mode);
            if (v != 0.0f) {
                const int e = route_id(ids, t * R + r, mode);
                route[n] = r;
                route_e[n] = e;
                route_w[n] = FMT == 0 ? v * factor[e] : v;
                ++n;
            }
        }
        active = n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int chunks = I / VPL, total = active * chunks;
    float acc[ROWS];
    for (int j = 0; j < ROWS; ++j) acc[j] = 0.0f;
    int a = tid / chunks, c = tid % chunks;  // (active route a, chunk c), advanced without division
    for (int p = tid; p < total; p += THREADS) {
        const int r = route[a];
        float xs[VPL];
        load_x<VPL>(act + ((long)t * R + r) * I + c * VPL, xs);
        const long row0 = (long)route_e[a] * H + h0;
        float part[ROWS];
        for (int j = 0; j < ROWS; ++j) part[j] = chunk_dot<FMT, VPL>(w, s, row0 + j, I, c, xs);
        const float rw = route_w[a];
        for (int j = 0; j < ROWS; ++j) acc[j] += rw * part[j];
        c += THREADS;
        while (c >= chunks) { c -= chunks; ++a; }
    }
    for (int j = 0; j < ROWS; ++j) acc[j] = simd_sum(acc[j]);
    if (lane == 0)
        for (int j = 0; j < ROWS; ++j) partial[sg][j] = acc[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < ROWS) {
        float v = 0.0f;
        for (int k = 0; k < SG; ++k) v += partial[k][tid];
        y[(long)t * H + h0 + tid] = T(v);
    }
}

// Gate/up + activation + down + routing sum in one dispatch (no gate_up -> down boundary). Grid: one threadgroup of
// SG simdgroups per (64-wide slice of the intermediate dim, pair), slice-major. A threadgroup
//   1. computes its 64 activations (gate rows i0.., up rows I + i0..) into threadgroup memory,
//   2. multiplies them by the matching 64 columns of down[e] for all H rows (4 lanes x 16 values per row),
//   3. adds wt * factor * that into a zeroed fp32 workspace [T, H] with atomics.
// The last threadgroup out (atomic ticket) converts the workspace to y and zeroes it and the ticket again, so it is
// clean for the next call (dispatches on the queue run one after another). The atomics make the summation order, so
// the last bit of y, vary between calls.
template <int FMT, typename T, int SG, int VPL>
kernel void experts_fused(device const uchar* gw, device const uchar* gs, device const float* gf, device const uchar* dw,
                          device const uchar* ds, device const float* df, device const T* x, device const uint* ids,
                          device const uchar* wt, device T* y, device atomic_float* ws, device atomic_uint* ticket,
                          constant int& H, constant int& I, constant int& P, constant int& R, constant int& mode,
                          constant float& gate_cap, constant float& up_cap, uint tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint ntg [[threadgroups_per_grid]]) {
    constexpr int SLICE = 64, THREADS = 32 * SG, LPR = SLICE / 16, ROWS_PER_STEP = THREADS / LPR;
    threadgroup float act[SLICE];
    const int p = tg % P, i0 = (tg / P) * SLICE, t = p / R;
    const float rw = route_wt(wt, p, mode);
    if (rw != 0.0f) {
        const int e = route_id(ids, p, mode);
        // 1. activations of the slice: simdgroup sg takes outputs i0 + sg, i0 + sg + SG, ...
        device const T* xt = x + (long)t * H;
        const int chunks = H / VPL;
        const float fg = FMT == 0 ? gf[2 * e] : 1.0f, fu = FMT == 0 ? gf[2 * e + 1] : 1.0f;
        for (int i = sg; i < SLICE; i += SG) {
            const long g_row = (long)e * 2 * I + i0 + i, u_row = g_row + I;
            float g = 0.0f, u = 0.0f;
            for (int c = lane; c < chunks; c += 32) {
                float xs[VPL];
                load_x<VPL>(xt + c * VPL, xs);
                g += chunk_dot<FMT, VPL>(gw, gs, g_row, H, c, xs);
                u += chunk_dot<FMT, VPL>(gw, gs, u_row, H, c, xs);
            }
            g = simd_sum(g);
            u = simd_sum(u);
            if (lane == 0) act[i] = activation(g * fg, u * fu, gate_cap, up_cap);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 2. down columns [i0, i0 + 64): lane `sub` of a row owns 16 consecutive activations (one decode chunk)
        const int sub = tid % LPR, c = (i0 + 16 * sub) / 16;
        float a[16];
        for (int j = 0; j < 16; ++j) a[j] = act[16 * sub + j];
        const float scale = FMT == 0 ? rw * df[e] : rw;
        device atomic_float* out = ws + (long)t * H;
        for (int h = tid / LPR; h < H; h += ROWS_PER_STEP) {
            float v = chunk_dot<FMT, 16>(dw, ds, (long)e * H + h, I, c, a);
            v += simd_shuffle_xor(v, 2);
            v += simd_shuffle_xor(v, 1);
            if (sub == 0) atomic_fetch_add_explicit(out + h, scale * v, memory_order_relaxed);
        }
    }
    // 3. the last threadgroup out writes y and cleans the workspace
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup bool last;
    if (tid == 0) last = atomic_fetch_add_explicit(ticket, 1u, memory_order_relaxed) == ntg - 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last) return;
    const int n = (P / R) * H;
    for (int k = tid; k < n; k += THREADS) y[k] = T(atomic_exchange_explicit(ws + k, 0.0f, memory_order_relaxed));
    if (tid == 0) atomic_store_explicit(ticket, 0u, memory_order_relaxed);
}

// ---- instantiations ----

#define GATE_UP(FMT, T, ROWS, VPL)                                                                                     \
    template [[host_name("gate_up_act_" #FMT "_" #T "_" #ROWS "_" #VPL)]] kernel void gate_up_act<FMT, T, ROWS, VPL>( \
        device const uchar*, device const uchar*, device const float*, device const T*, device const uint*,          \
        device const uchar*, device T*, constant int&, constant int&, constant int&, constant int&, constant int&,    \
        constant float&, constant float&, uint, uint);
#define DOWN(FMT, T, ROWS, VPL, SG)                                                                                    \
    template [[host_name("down_sum_" #FMT "_" #T "_" #ROWS "_" #VPL "_" #SG)]] kernel void                           \
    down_sum<FMT, T, ROWS, VPL, SG>(device const uchar*, device const uchar*, device const float*, device const T*,  \
        device const uint*, device const uchar*, device T*, constant int&, constant int&, constant int&,              \
        constant int&, constant int&, uint, uint, uint, uint);
#define FUSED(FMT, T, SG, VPL)                                                                                         \
    template [[host_name("experts_fused_" #FMT "_" #T "_" #SG "_" #VPL)]] kernel void experts_fused<FMT, T, SG, VPL>( \
        device const uchar*, device const uchar*, device const float*, device const uchar*, device const uchar*,     \
        device const float*, device const T*, device const uint*, device const uchar*, device T*,                    \
        device atomic_float*, device atomic_uint*, constant int&, constant int&, constant int&, constant int&,        \
        constant int&, constant float&, constant float&, uint, uint, uint, uint, uint);

#ifndef TUNE
// Production tiles (TILES in fp4_experts_opt.py)
#define PROD(T)                                                                                                        \
    GATE_UP(0, T, 1, 8) GATE_UP(1, T, 1, 8) DOWN(0, T, 4, 16, 4) DOWN(1, T, 2, 16, 4) FUSED(0, T, 8, 8) FUSED(1, T, 8, 8)
PROD(half) PROD(bfloat) PROD(float)
#else
// Tile sweep (bench.py --sweep): bfloat activations
#define SWEEP(FMT)                                                                                                     \
    GATE_UP(FMT, bfloat, 1, 8) GATE_UP(FMT, bfloat, 1, 16) GATE_UP(FMT, bfloat, 1, 32) GATE_UP(FMT, bfloat, 2, 8)     \
    GATE_UP(FMT, bfloat, 2, 16) GATE_UP(FMT, bfloat, 2, 32)                                                            \
    DOWN(FMT, bfloat, 2, 8, 4) DOWN(FMT, bfloat, 2, 16, 4) DOWN(FMT, bfloat, 2, 32, 4) DOWN(FMT, bfloat, 4, 8, 4)      \
    DOWN(FMT, bfloat, 4, 16, 4) DOWN(FMT, bfloat, 2, 16, 2) DOWN(FMT, bfloat, 2, 16, 8) DOWN(FMT, bfloat, 1, 16, 4)
SWEEP(0) SWEEP(1)
FUSED(0, bfloat, 4, 8) FUSED(0, bfloat, 8, 8) FUSED(0, bfloat, 8, 16) FUSED(0, bfloat, 16, 8) FUSED(1, bfloat, 4, 8) FUSED(1, bfloat, 8, 8) FUSED(1, bfloat, 8, 16)
FUSED(1, bfloat, 16, 8)
#endif

// part[p, h] = wt[p] * factor[e_p] * (down[e_p, h] . act[p]) for every (token, route) pair p (the caller sums the
// routes in a fixed order: deterministic). Threadgroup = (route, block of
// ROWS_TG rows); LPR lanes per row, a lane reuses its act chunk for the RPL rows it owns.
template <int FMT, typename T, int LPR, int RPL, int VPL, int SG>
kernel void down_route(device const uchar* w, device const uchar* s, device const float* factor, device const T* act,
                       device const uint* ids, device const uchar* wt, device float* part, constant int& H,
                       constant int& I, constant int& mode, uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int ROWS_SG = (32 / LPR) * RPL, ROWS_TG = ROWS_SG * SG;
    const int blocks = H / ROWS_TG, r = tg / blocks, h0 = (tg % blocks) * ROWS_TG + sg * ROWS_SG + (lane / LPR) * RPL;
    const float v = route_wt(wt, r, mode);
    const int t = lane % LPR;
    if (v == 0.0f) { if (t == 0) for (int j = 0; j < RPL; ++j) part[(long)r * H + h0 + j] = 0.0f; return; }
    const int e = route_id(ids, r, mode), chunks = I / VPL;
    float acc[RPL];
    for (int j = 0; j < RPL; ++j) acc[j] = 0.0f;
    device const T* a = act + (long)r * I;
    for (int c = t; c < chunks; c += LPR) {
        float xs[VPL];
        load_x<VPL>(a + c * VPL, xs);
        for (int j = 0; j < RPL; ++j) acc[j] += chunk_dot<FMT, VPL>(w, s, (long)e * H + h0 + j, I, c, xs);
    }
    for (int j = 0; j < RPL; ++j)
        for (int off = LPR / 2; off > 0; off /= 2) acc[j] += simd_shuffle_down(acc[j], off);
    if (t == 0) {
        const float f = FMT == 0 ? v * factor[e] : v;
        for (int j = 0; j < RPL; ++j) part[(long)r * H + h0 + j] = acc[j] * f;
    }
}
template [[host_name("down_route_0_bfloat")]] kernel void down_route<0, bfloat, 8, 1, 32, 4>(device const uchar*,
    device const uchar*, device const float*, device const bfloat*, device const uint*, device const uchar*, device float*,
    constant int&, constant int&, constant int&, uint, uint, uint);
template [[host_name("down_route_1_bfloat")]] kernel void down_route<1, bfloat, 8, 1, 32, 4>(device const uchar*,
    device const uchar*, device const float*, device const bfloat*, device const uint*, device const uchar*, device float*,
    constant int&, constant int&, constant int&, uint, uint, uint);
template [[host_name("down_route_0_half")]] kernel void down_route<0, half, 8, 1, 32, 4>(device const uchar*,
    device const uchar*, device const float*, device const half*, device const uint*, device const uchar*, device float*,
    constant int&, constant int&, constant int&, uint, uint, uint);
template [[host_name("down_route_1_half")]] kernel void down_route<1, half, 8, 1, 32, 4>(device const uchar*,
    device const uchar*, device const float*, device const half*, device const uint*, device const uchar*, device float*,
    constant int&, constant int&, constant int&, uint, uint, uint);
