// Batch-1 bf16 / fp16 matvec y[n] = W[n, :] . x for decode: LPR lanes per row, RPS rows per lane group, 8 values per
// lane per step (16-byte loads), fp32 accumulation; x chunk loaded once per step and reused across the RPS rows.
#include <metal_stdlib>
using namespace metal;
template <typename T, int LPR, int RPS, int NSG>
kernel void bmv(device const T* w, device const T* x, device T* y, constant int& N, constant int& K,
                uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
    constexpr int GROUPS = 32 / LPR, ROWS_SG = GROUPS * RPS;
    const int t = lane % LPR, row0 = (tg * NSG + sg) * ROWS_SG + (lane / LPR) * RPS;
    float acc[RPS] = {0.0f};
    for (int k = t * 8; k < K; k += LPR * 8) {
        const float4 x0 = float4(*(device const vec<T, 4>*)(x + k)), x1 = float4(*(device const vec<T, 4>*)(x + k + 4));
        for (int r = 0; r < RPS; ++r) {
            const long row = min(row0 + r, N - 1);
            device const vec<T, 4>* p = (device const vec<T, 4>*)(w + row * K + k);
            acc[r] += dot(float4(p[0]), x0) + dot(float4(p[1]), x1);
        }
    }
    for (int r = 0; r < RPS; ++r)
        for (int off = LPR / 2; off > 0; off /= 2) acc[r] += simd_shuffle_down(acc[r], off);
    if (t == 0)
        for (int r = 0; r < RPS; ++r)
            if (row0 + r < N) y[row0 + r] = T(acc[r]);
}
#define BMV(T, L, R, G) template [[host_name("bmv_" #T "_" #L "_" #R "_" #G)]] kernel void bmv<T, L, R, G>( \
    device const T*, device const T*, device T*, constant int&, constant int&, uint, uint, uint);
#define BMVS(L, R, G) BMV(bfloat, L, R, G) BMV(half, L, R, G)
BMVS(32, 1, 4) BMVS(32, 2, 4) BMVS(16, 1, 4) BMVS(16, 2, 4) BMVS(8, 1, 4) BMVS(8, 2, 4) BMVS(32, 4, 2) BMVS(16, 4, 2) BMVS(32, 1, 8) BMVS(16, 1, 8)
