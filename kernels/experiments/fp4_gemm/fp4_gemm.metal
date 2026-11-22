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
constant constexpr int BK = 64;
typedef bfloat TT;  // tile type: dequantized NVFP4 / MXFP8 values and bf16 activations are exact in bf16

template <int BM, typename T>
inline void load_a(device const T* a, threadgroup const int* row_src, int K, int k0, threadgroup TT* as, uint tid) {
    for (int u = tid; u < BM * BK / 8; u += 128) {  // 8 values per step
        const int m = u / (BK / 8), c = (u % (BK / 8)) * 8, src = row_src[m];
        if (src < 0) { for (int j = 0; j < 8; ++j) as[m * BK + c + j] = TT(0.0f); continue; }
        device const vec<T, 4>* p = (device const vec<T, 4>*)(a + (long)src * K + k0 + c);
        const float4 x0 = float4(p[0]), x1 = float4(p[1]);
        as[m * BK + c] = TT(x0.x); as[m * BK + c + 1] = TT(x0.y); as[m * BK + c + 2] = TT(x0.z); as[m * BK + c + 3] = TT(x0.w);
        as[m * BK + c + 4] = TT(x1.x); as[m * BK + c + 5] = TT(x1.y); as[m * BK + c + 6] = TT(x1.z); as[m * BK + c + 7] = TT(x1.w);
    }
}

// act_sorted[row, i] = activation(gate, up) for the sorted pairs; grid (tiles, I / 32), threadgroup 128
template <int FMT, typename T, int BM>
kernel void gemm_gate_up(device const uchar* w, device const uchar* s, device const float* factor, device const T* x,
                         device const long* perm, device const int* tile_e, device const int* tile_row,
                         device const int* row_end, device T* act, constant int& I, constant int& K, constant int& R,
                         constant float& gate_cap, constant float& up_cap,
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    const int e = tile_e[tg.x];
    if (e < 0) return;
    const int row0 = tile_row[tg.x], end = row_end[e], i0 = tg.y * 32;
    threadgroup TT smem[BM * BK + 64 * BK];  // A / W tiles in the k loop, then the fp32 output tile (same bytes)
    threadgroup TT* as = smem;
    threadgroup TT* ws = smem + BM * BK;
    threadgroup float* out_t = (threadgroup float*)smem;
    static_assert(BM * 64 * 4 <= (BM * BK + 64 * BK) * 2, "output tile must fit the staging buffers");
    threadgroup int row_src[BM];
    if (tid < BM) row_src[tid] = row0 + int(tid) < end ? int(perm[row0 + tid] / R) : -1;  // token of the pair
    constexpr int MB = BM / 32;  // 8-row blocks per simdgroup
    simdgroup_float8x8 acc[MB][8];
    for (int mb = 0; mb < MB; ++mb) for (int n = 0; n < 8; ++n) acc[mb][n] = simdgroup_float8x8(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = 0; k0 < K; k0 += BK) {
        load_a<BM, T>(x, row_src, K, k0, as, tid);
        for (int u = tid; u < 256; u += 128) {  // 64 rows (32 gate, 32 up) x 4 chunks of 16
            const int r = u >> 2, half_ = u & 3;
            const long row = (long)e * 2 * I + (r < 32 ? i0 + r : I + i0 + r - 32);
            float v[16];
            decode<FMT, 16>(w + row * row_bytes<FMT>(K), s + row * scale_bytes<FMT>(K), k0 / 16 + half_, v);
            for (int j = 0; j < 16; ++j) ws[r * BK + half_ * 16 + j] = TT(v[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_bfloat8x8 a[MB], b;
            for (int mb = 0; mb < MB; ++mb) simdgroup_load(a[mb], as + (sg * MB + mb) * 8 * BK + kk, BK);
            for (int n = 0; n < 8; ++n) {
                simdgroup_load(b, ws + n * 8 * BK + kk, BK, 0, true);
                for (int mb = 0; mb < MB; ++mb) simdgroup_multiply_accumulate(acc[mb][n], a[mb], b, acc[mb][n]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // epilogue: gate block n (0..3) / up block n + 4 -> ws as [32 rows][64 cols]
    for (int mb = 0; mb < MB; ++mb) for (int n = 0; n < 8; ++n) simdgroup_store(acc[mb][n], out_t + (sg * MB + mb) * 8 * 64 + n * 8, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float fg = FMT == 0 ? factor[2 * e] : 1.0f, fu = FMT == 0 ? factor[2 * e + 1] : 1.0f;
    for (int u = tid; u < BM * 32; u += 128) {
        const int m = u / 32, i = u % 32;
        if (row0 + m >= end) continue;
        const float g = out_t[m * 64 + i] * fg, up = out_t[m * 64 + 32 + i] * fu;
        act[(long)(row0 + m) * I + i0 + i] = T(activation(g, up, gate_cap, up_cap));
    }
}

// act_sorted[row, i] = activation(gate, up) for the sorted pairs; grid (tiles, I / 64), threadgroup 256 (8 simdgroups: 4 row blocks x 2 column halves)
template <int FMT, typename T, int BM>
kernel void gemm_gate_up_wide(device const uchar* w, device const uchar* s, device const float* factor, device const T* x,
                         device const long* perm, device const int* tile_e, device const int* tile_row,
                         device const int* row_end, device T* act, constant int& I, constant int& K, constant int& R,
                         constant float& gate_cap, constant float& up_cap,
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    const int e = tile_e[tg.x];
    if (e < 0) return;
    const int row0 = tile_row[tg.x], end = row_end[e], i0 = tg.y * 64;
    threadgroup TT smem[BM * BK + 128 * BK];
    threadgroup TT* as = smem;
    threadgroup TT* ws = smem + BM * BK;  // rows 0..63 gate i0.., 64..127 up I + i0..
    threadgroup float* out_t = (threadgroup float*)smem;
    static_assert(BM * 128 * 4 <= (BM * BK + 128 * BK) * 2, "output tile must fit the staging buffers");
    const uint msg = sg % 4, half_n = sg / 4;  // row block, column half (32 gate + 32 up columns)
    threadgroup int row_src[BM];
    if (tid < BM) row_src[tid] = row0 + int(tid) < end ? int(perm[row0 + tid] / R) : -1;  // token of the pair
    constexpr int MB = BM / 32;  // 8-row blocks per simdgroup
    simdgroup_float8x8 acc[MB][8];
    for (int mb = 0; mb < MB; ++mb) for (int n = 0; n < 8; ++n) acc[mb][n] = simdgroup_float8x8(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int u = tid; u < BM * BK / 8; u += 256) {
            const int m = u / (BK / 8), c = (u % (BK / 8)) * 8, src = row_src[m];
            if (src < 0) { for (int j = 0; j < 8; ++j) as[m * BK + c + j] = TT(0.0f); continue; }
            device const vec<T, 4>* p = (device const vec<T, 4>*)(x + (long)src * K + k0 + c);
            const float4 x0 = float4(p[0]), x1 = float4(p[1]);
            as[m * BK + c] = TT(x0.x); as[m * BK + c + 1] = TT(x0.y); as[m * BK + c + 2] = TT(x0.z); as[m * BK + c + 3] = TT(x0.w);
            as[m * BK + c + 4] = TT(x1.x); as[m * BK + c + 5] = TT(x1.y); as[m * BK + c + 6] = TT(x1.z); as[m * BK + c + 7] = TT(x1.w);
        }
        for (int u = tid; u < 512; u += 256) {  // 128 rows (64 gate, 64 up) x 4 chunks of 16
            const int r = u >> 2, half_ = u & 3;
            const long row = (long)e * 2 * I + (r < 64 ? i0 + r : I + i0 + r - 64);
            float v[16];
            decode<FMT, 16>(w + row * row_bytes<FMT>(K), s + row * scale_bytes<FMT>(K), k0 / 16 + half_, v);
            for (int j = 0; j < 16; ++j) ws[r * BK + half_ * 16 + j] = TT(v[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_bfloat8x8 a[MB], b;
            for (int mb = 0; mb < MB; ++mb) simdgroup_load(a[mb], as + (msg * MB + mb) * 8 * BK + kk, BK);
            for (int n = 0; n < 8; ++n) {
                const int wrow = (n < 4 ? half_n * 32 + n * 8 : 64 + half_n * 32 + (n - 4) * 8);
                simdgroup_load(b, ws + wrow * BK + kk, BK, 0, true);
                for (int mb = 0; mb < MB; ++mb) simdgroup_multiply_accumulate(acc[mb][n], a[mb], b, acc[mb][n]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // epilogue: gate block n (0..3) / up block n + 4 -> ws as [32 rows][64 cols]
    for (int mb = 0; mb < MB; ++mb)
        for (int n = 0; n < 8; ++n)  // out_t [BM rows][128]: gate 0..63, up 64..127
            simdgroup_store(acc[mb][n], out_t + (msg * MB + mb) * 8 * 128 + (n < 4 ? half_n * 32 + n * 8 : 64 + half_n * 32 + (n - 4) * 8), 128);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float fg = FMT == 0 ? factor[2 * e] : 1.0f, fu = FMT == 0 ? factor[2 * e + 1] : 1.0f;
    for (int u = tid; u < BM * 64; u += 256) {
        const int m = u / 64, i = u % 64;
        if (row0 + m >= end) continue;
        const float g = out_t[m * 128 + i] * fg, up = out_t[m * 128 + 64 + i] * fu;
        act[(long)(row0 + m) * I + i0 + i] = T(activation(g, up, gate_cap, up_cap));
    }
}

// y_sorted[row, h] = route weight * factor * (down[e, h] . act_sorted[row]); grid (tiles, H / 32), threadgroup 128
template <int FMT, typename T, int BM>
kernel void gemm_down(device const uchar* w, device const uchar* s, device const float* factor, device const T* act,
                      device const long* perm, device const uchar* wt, device const int* tile_e,
                      device const int* tile_row, device const int* row_end, device float* y, constant int& H,
                      constant int& I, constant int& mode, uint2 tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const int e = tile_e[tg.x];
    if (e < 0) return;
    const int row0 = tile_row[tg.x], end = row_end[e], h0 = tg.y * 32;
    threadgroup TT smem[BM * BK + 32 * BK];
    threadgroup TT* as = smem;
    threadgroup TT* ws = smem + BM * BK;
    threadgroup float* out_t = (threadgroup float*)smem;
    static_assert(BM * 32 * 4 <= (BM * BK + 32 * BK) * 2, "output tile must fit the staging buffers");
    threadgroup int row_src[BM];
    if (tid < BM) row_src[tid] = row0 + int(tid) < end ? row0 + int(tid) : -1;  // act is in sorted order
    constexpr int MB = BM / 32;
    simdgroup_float8x8 acc[MB][4];
    for (int mb = 0; mb < MB; ++mb) for (int n = 0; n < 4; ++n) acc[mb][n] = simdgroup_float8x8(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = 0; k0 < I; k0 += BK) {
        load_a<BM, T>(act, row_src, I, k0, as, tid);
        {  // 32 rows x 4 chunks of 16
            const int r = tid >> 2, half_ = tid & 3;
            const long row = (long)e * H + h0 + r;
            float v[16];
            decode<FMT, 16>(w + row * row_bytes<FMT>(I), s + row * scale_bytes<FMT>(I), k0 / 16 + half_, v);
            for (int j = 0; j < 16; ++j) ws[r * BK + half_ * 16 + j] = TT(v[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_bfloat8x8 a[MB], b;
            for (int mb = 0; mb < MB; ++mb) simdgroup_load(a[mb], as + (sg * MB + mb) * 8 * BK + kk, BK);
            for (int n = 0; n < 4; ++n) {
                simdgroup_load(b, ws + n * 8 * BK + kk, BK, 0, true);
                for (int mb = 0; mb < MB; ++mb) simdgroup_multiply_accumulate(acc[mb][n], a[mb], b, acc[mb][n]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int mb = 0; mb < MB; ++mb) for (int n = 0; n < 4; ++n) simdgroup_store(acc[mb][n], out_t + (sg * MB + mb) * 8 * 32 + n * 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float f = FMT == 0 ? factor[e] : 1.0f;
    for (int u = tid; u < BM * 32; u += 128) {
        const int m = u / 32, h = u % 32;
        if (row0 + m >= end) continue;
        const float rw = route_wt(wt, int(perm[row0 + m]), mode);
        y[(long)(row0 + m) * H + h0 + h] = out_t[m * 32 + h] * f * rw;
    }
}

#define GEMM_INST(FMT, T, TN, BMV)                                                                                       \
    template [[host_name("gemm_gate_up_" #FMT "_" #TN "_" #BMV)]] kernel void gemm_gate_up<FMT, T, BMV>(                \
        device const uchar*, device const uchar*, device const float*, device const T*, device const long*,             \
        device const int*, device const int*, device const int*, device T*, constant int&, constant int&,               \
        constant int&, constant float&, constant float&, uint2, uint, uint);                                             \
    template [[host_name("gemm_down_" #FMT "_" #TN "_" #BMV)]] kernel void gemm_down<FMT, T, BMV>(                      \
        device const uchar*, device const uchar*, device const float*, device const T*, device const long*,             \
        device const uchar*, device const int*, device const int*, device const int*, device float*, constant int&,     \
        constant int&, constant int&, uint2, uint, uint);
#define GEMM_ALL(BMV) GEMM_INST(0, bfloat, bfloat, BMV) GEMM_INST(0, half, half, BMV) GEMM_INST(1, bfloat, bfloat, BMV) GEMM_INST(1, half, half, BMV)
GEMM_ALL(32) GEMM_ALL(64)
#define GEMM_WIDE(FMT, T, TN, BMV) template [[host_name("gemm_gate_up_wide_" #FMT "_" #TN "_" #BMV)]] kernel void \
    gemm_gate_up_wide<FMT, T, BMV>(device const uchar*, device const uchar*, device const float*, device const T*, \
    device const long*, device const int*, device const int*, device const int*, device T*, constant int&, constant int&, \
    constant int&, constant float&, constant float&, uint2, uint, uint);
GEMM_WIDE(0, bfloat, bfloat, 32) GEMM_WIDE(1, bfloat, bfloat, 32)
GEMM_WIDE(0, half, half, 32) GEMM_WIDE(1, half, half, 32)
