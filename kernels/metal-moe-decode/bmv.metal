// Batch-1 bf16 / fp16 matvecs for exact (unquantized) decode: up to three independent y_i = W_i x_i (same K) in one
// dispatch. A simdgroup owns one row (32 lanes x 8 values = 16-byte loads), 8 simdgroups per threadgroup; fp32
// accumulation. Tuned on an M3 Max at Inkling-2-Mini's shapes: 1024 x 2304 43 -> 28.5 us vs torch's linear.
#include <metal_stdlib>
using namespace metal;
constant constexpr int NSG = 8;

template <typename T>
inline void bmv_rows(device const T* w, device const T* x, device T* y, int N, int K, int tg, uint sg, uint lane) {
    const int row = tg * NSG + int(sg);
    if (row >= N) return;
    device const T* wr = w + (long)row * K;
    float acc = 0.0f;
    for (int k = int(lane) * 8; k < K; k += 256) {
        const float4 x0 = float4(*(device const vec<T, 4>*)(x + k)), x1 = float4(*(device const vec<T, 4>*)(x + k + 4));
        acc += dot(float4(*(device const vec<T, 4>*)(wr + k)), x0) + dot(float4(*(device const vec<T, 4>*)(wr + k + 4)), x1);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = T(acc);
}

template <typename T>
kernel void bmv_multi(device const T* w0, device const T* x0, device T* y0, device const T* w1, device const T* x1,
                      device T* y1, device const T* w2, device const T* x2, device T* y2, constant int& N0,
                      constant int& N1, constant int& N2, constant int& T0, constant int& T1, constant int& K,
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    if (int(tg) < T0) bmv_rows<T>(w0, x0, y0, N0, K, tg, sg, lane);
    else if (int(tg) < T0 + T1) bmv_rows<T>(w1, x1, y1, N1, K, tg - T0, sg, lane);
    else bmv_rows<T>(w2, x2, y2, N2, K, tg - T0 - T1, sg, lane);
}
#define BMVM(T) template [[host_name("bmv_multi_" #T)]] kernel void bmv_multi<T>(device const T*, device const T*, \
    device T*, device const T*, device const T*, device T*, device const T*, device const T*, device T*, constant int&, \
    constant int&, constant int&, constant int&, constant int&, constant int&, uint, uint, uint);
BMVM(bfloat) BMVM(half)
