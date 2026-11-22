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


// ---------------------------------------------------------------------------------------------------------------------
// Prefill (T >> 1): grouped GEMMs over the (token, route) pairs sorted by expert. A tile = BM = 32 consecutive sorted
// pairs of one expert (`tile_e`, `tile_row` from the host), 4 simdgroups, simdgroup s owns rows 8s..8s+7; weights are
// dequantized per 32-wide k slice into threadgroup memory (fp32), never to DRAM; fp32 simdgroup MMA.
constant constexpr int BM = 32, BK = 32;

template <typename T>
inline void load_a(device const T* a, threadgroup const int* row_src, int K, int k0, threadgroup float* as, uint tid) {
    for (int u = tid; u < BM * BK / 8; u += 128) {  // 8 values per step
        const int m = u / (BK / 8), c = (u % (BK / 8)) * 8, src = row_src[m];
        if (src < 0) { for (int j = 0; j < 8; ++j) as[m * BK + c + j] = 0.0f; continue; }
        device const vec<T, 4>* p = (device const vec<T, 4>*)(a + (long)src * K + k0 + c);
        const float4 x0 = float4(p[0]), x1 = float4(p[1]);
        as[m * BK + c] = x0.x; as[m * BK + c + 1] = x0.y; as[m * BK + c + 2] = x0.z; as[m * BK + c + 3] = x0.w;
        as[m * BK + c + 4] = x1.x; as[m * BK + c + 5] = x1.y; as[m * BK + c + 6] = x1.z; as[m * BK + c + 7] = x1.w;
    }
}

// act_sorted[row, i] = activation(gate, up) for the sorted pairs; grid (tiles, I / 32), threadgroup 128
template <int FMT, typename T>
kernel void gemm_gate_up(device const uchar* w, device const uchar* s, device const float* factor, device const T* x,
                         device const long* perm, device const int* tile_e, device const int* tile_row,
                         device const int* row_end, device T* act, constant int& I, constant int& K, constant int& R,
                         constant float& gate_cap, constant float& up_cap,
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    const int e = tile_e[tg.x];
    if (e < 0) return;
    const int row0 = tile_row[tg.x], end = row_end[e], i0 = tg.y * 32;
    threadgroup float as[BM * BK], ws[64 * BK];
    threadgroup int row_src[BM];
    if (tid < BM) row_src[tid] = row0 + int(tid) < end ? int(perm[row0 + tid] / R) : -1;  // token of the pair
    simdgroup_float8x8 acc[8];
    for (int n = 0; n < 8; ++n) acc[n] = simdgroup_float8x8(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = 0; k0 < K; k0 += BK) {
        load_a<T>(x, row_src, K, k0, as, tid);
        for (int u = tid; u < 128; u += 128) {  // 64 rows (32 gate, 32 up) x 2 halves
            const int r = u >> 1, half_ = u & 1;
            const long row = (long)e * 2 * I + (r < 32 ? i0 + r : I + i0 + r - 32);
            float v[16];
            decode<FMT, 16>(w + row * row_bytes<FMT>(K), s + row * scale_bytes<FMT>(K), k0 / 16 + half_, v);
            for (int j = 0; j < 16; ++j) ws[r * BK + half_ * 16 + j] = v[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, as + sg * 8 * BK + kk, BK);
            for (int n = 0; n < 8; ++n) {
                simdgroup_load(b, ws + n * 8 * BK + kk, BK, 0, true);
                simdgroup_multiply_accumulate(acc[n], a, b, acc[n]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // epilogue: gate block n (0..3) / up block n + 4 -> ws as [32 rows][64 cols]
    for (int n = 0; n < 8; ++n) simdgroup_store(acc[n], ws + sg * 8 * 64 + n * 8, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float fg = FMT == 0 ? factor[2 * e] : 1.0f, fu = FMT == 0 ? factor[2 * e + 1] : 1.0f;
    for (int u = tid; u < BM * 32; u += 128) {
        const int m = u / 32, i = u % 32;
        if (row0 + m >= end) continue;
        const float g = ws[m * 64 + i] * fg, up = ws[m * 64 + 32 + i] * fu;
        act[(long)(row0 + m) * I + i0 + i] = T(activation(g, up, gate_cap, up_cap));
    }
}

// y_sorted[row, h] = route weight * factor * (down[e, h] . act_sorted[row]); grid (tiles, H / 32), threadgroup 128
template <int FMT, typename T>
kernel void gemm_down(device const uchar* w, device const uchar* s, device const float* factor, device const T* act,
                      device const long* perm, device const uchar* wt, device const int* tile_e,
                      device const int* tile_row, device const int* row_end, device float* y, constant int& H,
                      constant int& I, constant int& mode, uint2 tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const int e = tile_e[tg.x];
    if (e < 0) return;
    const int row0 = tile_row[tg.x], end = row_end[e], h0 = tg.y * 32;
    threadgroup float as[BM * BK], ws[32 * BK];
    threadgroup int row_src[BM];
    if (tid < BM) row_src[tid] = row0 + int(tid) < end ? row0 + int(tid) : -1;  // act is in sorted order
    simdgroup_float8x8 acc[4];
    for (int n = 0; n < 4; ++n) acc[n] = simdgroup_float8x8(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = 0; k0 < I; k0 += BK) {
        load_a<T>(act, row_src, I, k0, as, tid);
        if (tid < 64) {
            const int r = tid >> 1, half_ = tid & 1;
            const long row = (long)e * H + h0 + r;
            float v[16];
            decode<FMT, 16>(w + row * row_bytes<FMT>(I), s + row * scale_bytes<FMT>(I), k0 / 16 + half_, v);
            for (int j = 0; j < 16; ++j) ws[r * BK + half_ * 16 + j] = v[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, as + sg * 8 * BK + kk, BK);
            for (int n = 0; n < 4; ++n) {
                simdgroup_load(b, ws + n * 8 * BK + kk, BK, 0, true);
                simdgroup_multiply_accumulate(acc[n], a, b, acc[n]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int n = 0; n < 4; ++n) simdgroup_store(acc[n], as + sg * 8 * 32 + n * 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float f = FMT == 0 ? factor[e] : 1.0f;
    for (int u = tid; u < BM * 32; u += 128) {
        const int m = u / 32, h = u % 32;
        if (row0 + m >= end) continue;
        const float rw = route_wt(wt, int(perm[row0 + m]), mode);
        y[(long)(row0 + m) * H + h0 + h] = as[m * 32 + h] * f * rw;
    }
}

#define GEMM_INST(FMT, T, TN)                                                                                            \
    template [[host_name("gemm_gate_up_" #FMT "_" #TN)]] kernel void gemm_gate_up<FMT, T>(                              \
        device const uchar*, device const uchar*, device const float*, device const T*, device const long*,             \
        device const int*, device const int*, device const int*, device T*, constant int&, constant int&,               \
        constant int&, constant float&, constant float&, uint2, uint, uint);                                             \
    template [[host_name("gemm_down_" #FMT "_" #TN)]] kernel void gemm_down<FMT, T>(                                    \
        device const uchar*, device const uchar*, device const float*, device const T*, device const long*,             \
        device const uchar*, device const int*, device const int*, device const int*, device float*, constant int&,     \
        constant int&, constant int&, uint2, uint, uint);
GEMM_INST(0, bfloat, bfloat) GEMM_INST(0, half, half) GEMM_INST(0, float, float)
GEMM_INST(1, bfloat, bfloat) GEMM_INST(1, half, half) GEMM_INST(1, float, float)
