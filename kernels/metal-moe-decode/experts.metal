// Batch-1 MoE expert decode kernels for MLX affine 4-bit banks (one token, R top-k routes).
//
// Layout (MLX packing): a bank is [E, N, K / 8] uint32 (8 nibbles per word, low nibble first), with scales and biases
// [E, N, K / GS] in the activation dtype: w[n, k] = scale[n, k / GS] * q[n, k] + bias[n, k / GS].
//
// All kernels are pure streaming reads of the routed experts' weights; the activations (x, the [R, I] SiLU product)
// are a few KB and stay in cache. What matters for bandwidth is the number of independent 4-bit loads each lane has in
// flight, so every lane owns several weight rows that share one activation chunk held in registers.
//
//   experts_decode_fused  the whole token in one dispatch (experimental): a threadgroup per (route, 64-wide slice of
//                         the intermediate dim) computes its slice of SiLU(gate) * up, multiplies it by the matching
//                         down columns and adds the result into an fp32 workspace with atomics; the last threadgroup
//                         writes y. Ties gate_up + down_sum with all routes active; slower on masked passes.
//   experts_gate_up       one simdgroup = one route x ROWS gate rows + the ROWS matching up rows (gate row i and up row
//                         I + i), so SiLU(gate) * up is applied in-register and only the [R, I] product is written.
//                         The raw variant (SWIGLU = false) writes both rows for callers that apply their own activation.
//   experts_down_sum      one threadgroup = ROWS output rows; its threads stride over the (active route, chunk) pairs
//                         of the concatenated [R * I] input so every thread has work whatever I is, and the routing-
//                         weighted sum happens in the accumulator.
//
// Zero-weight routes (streamed-experts sentinels) read nothing in every kernel, so a correction pass over the 1-2
// routes that missed a cache costs only those routes.
//
// Dequantization trick (as in MLX's qmv): the chunk's activations are pre-scaled by 16^-j for nibble j so a nibble is
// read with a mask, no shift: sum_j x_j * q_j == sum_j (x_j * 16^-j) * float(word & (0xF << 4j)), exact in fp32.

#include <metal_stdlib>
using namespace metal;

// VPL (values per lane per step) activations of one chunk, pre-scaled for masked nibble reads, plus their plain sum
// (for the bias term). VPL is 8 (one uint32 per row), 16 (a uint2) or 32 (a uint4).
template <int VPL>
struct Chunk {
    float x[VPL];
    float sum;
};

template <int VPL, typename T>
inline Chunk<VPL> load_chunk(device const T* src) {
    Chunk<VPL> c;
    c.sum = 0.0f;
    device const vec<T, 4>* v = (device const vec<T, 4>*)src;
    for (int i = 0; i < VPL / 4; ++i) {
        float4 f = float4(v[i]);
        c.sum += f.x + f.y + f.z + f.w;
        // nibble positions 4i..4i+3 within its word: shifts 0/4/8/12 or 16/20/24/28
        float s = (i & 1) ? 1.0f / 65536.0f : 1.0f;
        c.x[4 * i + 0] = f.x * s;
        c.x[4 * i + 1] = f.y * s * (1.0f / 16.0f);
        c.x[4 * i + 2] = f.z * s * (1.0f / 256.0f);
        c.x[4 * i + 3] = f.w * s * (1.0f / 4096.0f);
    }
    return c;
}

// sum_j q_j * x_j over one uint32 of 8 nibbles against pre-scaled activations xs[0..7].
inline float qdot8(uint q, thread const float* xs) {
    return xs[0] * float(q & 0x0000000Fu) + xs[1] * float(q & 0x000000F0u) + xs[2] * float(q & 0x00000F00u) +
           xs[3] * float(q & 0x0000F000u) + xs[4] * float(q & 0x000F0000u) + xs[5] * float(q & 0x00F00000u) +
           xs[6] * float(q & 0x0F000000u) + xs[7] * float(q & 0xF0000000u);
}

// One row's contribution for one chunk: scale * (q . x) + bias * sum(x). `wq` points at the row's chunk words.
template <int VPL, typename T>
inline float chunk_dot(device const uint* wq, T scale, T bias, thread const Chunk<VPL>& c) {
    float d = 0.0f;
    if (VPL == 8) {
        d = qdot8(wq[0], c.x);
    } else if (VPL == 16) {
        uint2 q = *(device const uint2*)wq;
        d = qdot8(q.x, c.x) + qdot8(q.y, c.x + 8);
    } else {
        uint4 q = *(device const uint4*)wq;
        d = qdot8(q.x, c.x) + qdot8(q.y, c.x + 8) + qdot8(q.z, c.x + 16) + qdot8(q.w, c.x + 24);
    }
    return float(scale) * d + float(bias) * c.sum;
}

inline float silu(float v) { return v / (1.0f + metal::exp(-v)); }

// Gathered gate/up for one token: route r = ids[r]'s rows of a [E, 2I, K / 8] bank. Grid: one simdgroup per
// (route, block of ROWS gate rows); routes with a zero weight write zeros and read nothing.
//   SWIGLU: out [R, I] (TO) = SiLU(gate) * up
//   raw:    out [R, 2I] (TO) = gate | up
template <typename T, typename TO, int GS, int ROWS, int VPL, bool SWIGLU>
kernel void experts_gate_up(device const uint* w, device const T* sc, device const T* bi, device const T* x,
                            device const int* ids, device const float* wt, device TO* out, constant int& I,
                            constant int& K, constant int& R, uint gid [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const int blocks = I / ROWS;
    const int s = gid / 32;
    const int r = s / blocks, i0 = (s % blocks) * ROWS;
    if (r >= R) return;
    if (wt[r] == 0.0f) {  // masked route (streamed-experts sentinel)
        if (lane < ROWS) {
            if (SWIGLU) {
                out[r * I + i0 + lane] = TO(0);
            } else {
                out[r * 2 * I + i0 + lane] = TO(0);
                out[r * 2 * I + I + i0 + lane] = TO(0);
            }
        }
        return;
    }
    const int wpr = K / 8, gpr = K / GS, chunks = K / VPL;
    const long gate_row = (long)ids[r] * 2 * I + i0, up_row = gate_row + I;

    float acc[2 * ROWS];
    for (int j = 0; j < 2 * ROWS; ++j) acc[j] = 0.0f;
    for (int c = lane; c < chunks; c += 32) {
        Chunk<VPL> xc = load_chunk<VPL>(x + c * VPL);
        const int word = c * (VPL / 8), g = c * VPL / GS;
        for (int j = 0; j < ROWS; ++j) {
            long gr = gate_row + j, ur = up_row + j;
            acc[j] += chunk_dot<VPL>(w + gr * wpr + word, sc[gr * gpr + g], bi[gr * gpr + g], xc);
            acc[ROWS + j] += chunk_dot<VPL>(w + ur * wpr + word, sc[ur * gpr + g], bi[ur * gpr + g], xc);
        }
    }
    for (int j = 0; j < 2 * ROWS; ++j) acc[j] = simd_sum(acc[j]);
    if (lane < ROWS) {
        // every lane holds the sums; lane j writes row i0 + j (registers indexed by a constant after unrolling)
        float gate = 0.0f, up = 0.0f;
        for (int j = 0; j < ROWS; ++j) {
            if (lane == uint(j)) {
                gate = acc[j];
                up = acc[ROWS + j];
            }
        }
        if (SWIGLU) {
            out[r * I + i0 + lane] = TO(silu(gate) * up);
        } else {
            out[r * 2 * I + i0 + lane] = TO(gate);
            out[r * 2 * I + I + i0 + lane] = TO(up);
        }
    }
}

// Down projections + routing sum for one token: y[h] = sum_r wt[r] * (down[ids[r], h] . act[r]), `down` a
// [E, H, I / 8] bank, `act` [R, I] (TA). Grid: one threadgroup of SG simdgroups per ROWS output rows; its threads
// stride over the (active route, chunk) pairs together and their partial sums are reduced in threadgroup memory.
template <typename T, typename TA, int GS, int ROWS, int VPL, int SG>
kernel void experts_down_sum(device const uint* w, device const T* sc, device const T* bi, device const TA* act,
                             device const int* ids, device const float* wt, device T* y, constant int& H,
                             constant int& I, constant int& R, uint tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int MAX_ROUTES = 64, THREADS = 32 * SG;
    threadgroup int route[MAX_ROUTES];
    threadgroup float route_wt[MAX_ROUTES];
    threadgroup int active;
    threadgroup float partial[SG][ROWS];
    if (tid == 0) {  // compact the nonzero-weight routes: masked ones cost nothing below
        int n = 0;
        for (int r = 0; r < R; ++r) {
            float v = wt[r];
            if (v != 0.0f) {
                route[n] = r;
                route_wt[n] = v;
                ++n;
            }
        }
        active = n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int h0 = tg * ROWS;  // the launcher sizes the grid so that H % ROWS == 0
    const int wpr = I / 8, gpr = I / GS, chunks = I / VPL, total = active * chunks;

    float acc[ROWS];
    for (int j = 0; j < ROWS; ++j) acc[j] = 0.0f;
    // (active route a, chunk c) pairs, p = a * chunks + c, advanced incrementally (no division in the loop)
    int a = tid / chunks, c = tid % chunks;
    for (int p = tid; p < total; p += THREADS) {
        const int r = route[a];
        Chunk<VPL> xc = load_chunk<VPL>(act + r * I + c * VPL);
        const long row0 = (long)ids[r] * H + h0;
        const int word = c * (VPL / 8), g = c * VPL / GS;
        float part[ROWS];
        for (int j = 0; j < ROWS; ++j) {
            long row = row0 + j;
            part[j] = chunk_dot<VPL>(w + row * wpr + word, sc[row * gpr + g], bi[row * gpr + g], xc);
        }
        const float rw = route_wt[a];
        for (int j = 0; j < ROWS; ++j) acc[j] += rw * part[j];
        c += THREADS;
        while (c >= chunks) {
            c -= chunks;
            ++a;
        }
    }
    for (int j = 0; j < ROWS; ++j) acc[j] = simd_sum(acc[j]);
    if (SG == 1) {
        if (lane < ROWS) {
            float v = 0.0f;
            for (int j = 0; j < ROWS; ++j)
                if (lane == uint(j)) v = acc[j];
            y[h0 + lane] = T(v);
        }
        return;
    }
    if (lane == 0)
        for (int j = 0; j < ROWS; ++j) partial[sg][j] = acc[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < ROWS) {
        float v = 0.0f;
        for (int s = 0; s < SG; ++s) v += partial[s][tid];
        y[h0 + tid] = T(v);
    }
}

// The whole token in one dispatch: y = sum_r wt[r] * down[e_r] @ (SiLU(gate[e_r] @ x) * (up[e_r] @ x)).
// Grid: one threadgroup of SG simdgroups per (route, SLICE-wide slice of the intermediate dim). A threadgroup
//   1. computes its slice of SiLU(gate) * up (2 * SLICE rows of the gate/up bank) into threadgroup memory,
//   2. multiplies it by the matching SLICE columns of down[e_r] for all H rows, LPR lanes per row,
//   3. adds wt[r] * that into a zeroed fp32 workspace with atomics.
// The last threadgroup to finish (atomic ticket) converts the workspace to y and re-zeroes it and the ticket, so the
// workspace is clean for the next call (dispatches on a queue are serialized).
template <typename T, int GS, int SLICE, int SG, int VPL>
kernel void experts_decode_fused(device const uint* gu, device const T* gs, device const T* gb, device const uint* dn,
                                 device const T* ds, device const T* db, device const T* x, device const int* ids,
                                 device const float* wt, device T* y, device atomic_float* ws,
                                 device atomic_uint* ticket, constant int& H, constant int& I, constant int& R,
                                 uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint ntg [[threadgroups_per_grid]]) {
    constexpr int THREADS = 32 * SG;
    threadgroup float act[SLICE];
    const int slices = I / SLICE;
    const int r = tg / slices, i0 = (tg % slices) * SLICE;
    const float rw = wt[r];

    if (rw != 0.0f) {
        const int e = ids[r];
        // 1. gate/up: simdgroup sg handles gate rows i0 + sg, i0 + sg + SG, ... and their up rows
        {
            const int K = H, wpr = K / 8, gpr = K / GS, chunks = K / VPL;
            for (int i = sg; i < SLICE; i += SG) {
                const long g_row = (long)e * 2 * I + i0 + i, u_row = g_row + I;
                float ga = 0.0f, ua = 0.0f;
                for (int c = lane; c < chunks; c += 32) {
                    Chunk<VPL> xc = load_chunk<VPL>(x + c * VPL);
                    const int word = c * (VPL / 8), g = c * VPL / GS;
                    ga += chunk_dot<VPL>(gu + g_row * wpr + word, gs[g_row * gpr + g], gb[g_row * gpr + g], xc);
                    ua += chunk_dot<VPL>(gu + u_row * wpr + word, gs[u_row * gpr + g], gb[u_row * gpr + g], xc);
                }
                ga = simd_sum(ga);
                ua = simd_sum(ua);
                if (lane == 0) act[i] = silu(ga) * ua;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 2. down columns [i0, i0 + SLICE): LPR lanes per row, each owning SLICE / LPR consecutive inputs (one group)
        {
            constexpr int LPR = SLICE / 16, ROWS_PER_STEP = THREADS / LPR;  // 16 inputs (a uint2) per lane
            const int wpr = I / 8, gpr = I / GS;
            const int sub = tid % LPR, k0 = sub * 16;
            Chunk<16> ac;
            ac.sum = 0.0f;
            for (int j = 0; j < 16; ++j) {
                float v = act[k0 + j];
                ac.sum += v;
                ac.x[j] = v * (1.0f / float(1u << (4 * (j % 8))));
            }
            const int word = (i0 + k0) / 8, g = (i0 + k0) / GS;
            for (int h = tid / LPR; h < H; h += ROWS_PER_STEP) {
                const long row = (long)e * H + h;
                float v = chunk_dot<16>(dn + row * wpr + word, ds[row * gpr + g], db[row * gpr + g], ac);
                for (int off = LPR / 2; off > 0; off /= 2) v += simd_shuffle_xor(v, off);
                if (sub == 0) atomic_fetch_add_explicit(ws + h, rw * v, memory_order_relaxed);
            }
        }
    }

    // 3. last threadgroup out: workspace -> y, then clean up for the next call
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup uint last;
    if (tid == 0) last = atomic_fetch_add_explicit(ticket, 1u, memory_order_relaxed) == ntg - 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last) return;
    for (int h = tid; h < H; h += THREADS) {
        y[h] = T(atomic_exchange_explicit(ws + h, 0.0f, memory_order_relaxed));
    }
    if (tid == 0) atomic_store_explicit(ticket, 0u, memory_order_relaxed);
}

// ---- instantiations: dtype T in {float, half, bfloat}, GS in {32, 64, 128}; tile shapes picked by the launcher ----

#define GATE_UP(T, TO, TON, GS, ROWS, VPL, SW, SWN)                                                                    \
    template [[host_name("experts_gate_up_" SWN "_" #T "_" TON "_" #GS "_" #ROWS "_" #VPL)]] kernel void             \
    experts_gate_up<T, TO, GS, ROWS, VPL, SW>(device const uint*, device const T*, device const T*, device const T*, \
                                              device const int*, device const float*, device TO*, constant int&,     \
                                              constant int&, constant int&, uint, uint);

#define DOWN(T, TA, TAN, GS, ROWS, VPL, SG)                                                                            \
    template [[host_name("experts_down_sum_" #T "_" TAN "_" #GS "_" #ROWS "_" #VPL "_" #SG)]] kernel void             \
    experts_down_sum<T, TA, GS, ROWS, VPL, SG>(device const uint*, device const T*, device const T*,                 \
                                               device const TA*, device const int*, device const float*, device T*,  \
                                               constant int&, constant int&, constant int&, uint, uint, uint, uint);

#define FUSED(T, GS, SLICE, SG, VPL)                                                                                  \
    template [[host_name("experts_decode_fused_" #T "_" #GS "_" #SLICE "_" #SG "_" #VPL)]] kernel void              \
    experts_decode_fused<T, GS, SLICE, SG, VPL>(                                                                     \
        device const uint*, device const T*, device const T*, device const uint*, device const T*, device const T*,  \
        device const T*, device const int*, device const float*, device T*, device atomic_float*, device atomic_uint*, \
        constant int&, constant int&, constant int&, uint, uint, uint, uint, uint);

// Production tiles (GATE_UP_TILE / DOWN_TILE / FUSED_TILE in experts.py)
#define PROD_ACT(T, GS)                                                                                                \
    GATE_UP(T, T, "act", GS, 1, 8, true, "swiglu") GATE_UP(T, T, "act", GS, 1, 16, true, "swiglu")                    \
    GATE_UP(T, T, "act", GS, 1, 8, false, "raw") GATE_UP(T, T, "act", GS, 1, 16, false, "raw")                         \
    DOWN(T, T, "act", GS, 2, 8, 4) DOWN(T, T, "act", GS, 2, 16, 4) FUSED(T, GS, 64, 8, 8) FUSED(T, GS, 64, 8, 16)
#define PROD_F32(T, GS)                                                                                                \
    GATE_UP(T, float, "f32", GS, 1, 8, true, "swiglu") GATE_UP(T, float, "f32", GS, 1, 16, true, "swiglu")            \
    DOWN(T, float, "f32", GS, 2, 8, 4) DOWN(T, float, "f32", GS, 2, 16, 4)
#define PROD(GS) PROD_ACT(float, GS) PROD_ACT(half, GS) PROD_ACT(bfloat, GS) PROD_F32(half, GS) PROD_F32(bfloat, GS)
PROD(32)
PROD(64)
PROD(128)

#if EXPERTS_TUNE
// Tile sweep (bench.py --sweep): bfloat weights, group 64, fp32 intermediates
#define GU_SWEEP(ROWS)                                                                                                 \
    GATE_UP(bfloat, float, "f32", 64, ROWS, 8, true, "swiglu") GATE_UP(bfloat, float, "f32", 64, ROWS, 16, true, "swiglu") \
    GATE_UP(bfloat, float, "f32", 64, ROWS, 32, true, "swiglu")
#define DOWN_SWEEP(ROWS, SG)                                                                                           \
    DOWN(bfloat, float, "f32", 64, ROWS, 8, SG) DOWN(bfloat, float, "f32", 64, ROWS, 16, SG)                          \
    DOWN(bfloat, float, "f32", 64, ROWS, 32, SG)
GU_SWEEP(2) GU_SWEEP(4) GATE_UP(bfloat, float, "f32", 64, 1, 32, true, "swiglu")
DOWN_SWEEP(2, 2) DOWN(bfloat, float, "f32", 64, 2, 32, 4) DOWN_SWEEP(2, 8) DOWN_SWEEP(4, 2) DOWN_SWEEP(4, 4) DOWN_SWEEP(4, 8)
DOWN_SWEEP(8, 2) DOWN_SWEEP(8, 4) DOWN_SWEEP(8, 8) DOWN_SWEEP(16, 2) DOWN_SWEEP(16, 4) DOWN_SWEEP(16, 8)
FUSED(bfloat, 64, 64, 4, 8) FUSED(bfloat, 64, 64, 16, 8) FUSED(bfloat, 64, 64, 4, 16) FUSED(bfloat, 64, 64, 16, 16)
FUSED(bfloat, 64, 32, 4, 8) FUSED(bfloat, 64, 32, 8, 8) FUSED(bfloat, 64, 32, 4, 16) FUSED(bfloat, 64, 32, 8, 16)
#endif
