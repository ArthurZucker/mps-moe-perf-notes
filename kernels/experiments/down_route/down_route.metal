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


// part[r, h] = wt[r] * factor[e_r] * (down[e_r, h] . act[r]) for one token's R routes. Threadgroup = (route, block of
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
#define DR(FMT, LPR, RPL, VPL, SG) template [[host_name("down_route_" #FMT "_" #LPR "_" #RPL "_" #VPL "_" #SG)]] kernel void \
  down_route<FMT, bfloat, LPR, RPL, VPL, SG>(device const uchar*, device const uchar*, device const float*, device const bfloat*, \
  device const uint*, device const uchar*, device float*, constant int&, constant int&, constant int&, uint, uint, uint);
#define DRF(LPR, RPL, VPL, SG) DR(0, LPR, RPL, VPL, SG) DR(1, LPR, RPL, VPL, SG)
DRF(8, 1, 16, 4) DRF(8, 2, 16, 4) DRF(8, 4, 16, 4) DRF(16, 1, 16, 4) DRF(16, 2, 16, 4) DRF(32, 1, 16, 4) DRF(4, 1, 16, 4)
DRF(8, 2, 16, 8) DRF(8, 1, 32, 4) DRF(4, 2, 32, 4) DRF(8, 2, 8, 4)
