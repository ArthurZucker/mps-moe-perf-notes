#include <metal_stdlib>
using namespace metal;

// Batch-1 4-bit affine matvec (MLX packing): y[n] = sum_k (scale[n, g] * q[n, k] + bias[n, g]) * x[k], g = k / GS.
// W: [N, K / 8] uint32, 8 nibbles per word, low nibble first. scales / biases: [N, K / GS]. x: [K]. y: [N].
//
// Work layout (all compile-time, one instantiation per shape class):
//   - LPR lanes cooperate on a row group; a simdgroup holds 32 / LPR row groups.
//   - each row group owns ROWS consecutive rows: a lane loads its x chunk once and reuses it for all ROWS rows, so
//     x traffic (bf16, 4x the bytes of the nibbles it multiplies) is divided by ROWS.
//   - each lane takes VPL consecutive values (VPL / 8 words: one uint, uint2 or uint4 load per row) per step.
//   - SPLIT simdgroups of a threadgroup share the same rows and interleave over K (split-K inside the threadgroup,
//     reduced through threadgroup memory): rows-poor, K-long matrices still get enough loads in flight.
//   - a threadgroup holds NSG simdgroups, i.e. NSG / SPLIT row sets.
//
// The dequantization uses shift-free masks: nibble j of a word is (word & (0xF << 4j)) = q * 16^j, so x is pre-scaled
// by 16^-j once per chunk (exact in fp32, shared across the ROWS rows) and each nibble costs an and + convert + fma.
// sum_k bias * x[k] = bias * sum(x chunk), also shared across rows. Accumulation is fp32.

template <int WORDS>
struct Words;
template <>
struct Words<1> {
    static inline void load(device const uint32_t* p, thread uint32_t* q) { q[0] = p[0]; }
};
template <>
struct Words<2> {
    static inline void load(device const uint32_t* p, thread uint32_t* q) {
        uint2 v = *(device const uint2*)p;
        q[0] = v.x, q[1] = v.y;
    }
};
template <>
struct Words<4> {
    static inline void load(device const uint32_t* p, thread uint32_t* q) {
        uint4 v = *(device const uint4*)p;
        q[0] = v.x, q[1] = v.y, q[2] = v.z, q[3] = v.w;
    }
};

template <typename T, int GS, int LPR, int ROWS, int VPL, int SPLIT, int NSG>
inline void qmv4_body(device const uint32_t* w, device const T* sc, device const T* bi, device const T* x, device T* y,
                      int N, int K, uint tg, uint sg, uint lane, threadgroup float* partial_) {
    static_assert(32 % LPR == 0 && NSG % SPLIT == 0 && GS % VPL == 0 && VPL % 8 == 0, "bad qmv4 configuration");
    constexpr int WORDS = VPL / 8;
    constexpr int GROUPS_PER_SG = 32 / LPR;
    constexpr int ROWS_PER_SG = GROUPS_PER_SG * ROWS;
    constexpr int ROWS_PER_TG = NSG / SPLIT * ROWS_PER_SG;
    constexpr int STEP = SPLIT * LPR * VPL;

    const int ks = sg % SPLIT, t = lane % LPR;
    const int row0 = tg * ROWS_PER_TG + (sg / SPLIT) * ROWS_PER_SG + (lane / LPR) * ROWS;
    const int words_per_row = K / 8, groups = K / GS;

    // Rows past N are clamped onto the last row: same loads, no branch in the loop, never written.
    device const uint32_t* wr[ROWS];
    device const T* sr[ROWS];
    device const T* br[ROWS];
    for (int r = 0; r < ROWS; ++r) {
        long row = min(row0 + r, N - 1);
        wr[r] = w + row * words_per_row;
        sr[r] = sc + row * groups;
        br[r] = bi + row * groups;
    }

    float acc[ROWS] = {0.0f};
    for (int k = (ks * LPR + t) * VPL; k < K; k += STEP) {
        // x chunk, pre-scaled per nibble position, and its sum (for the bias term).
        float xs[VPL];
        float xsum = 0.0f;
        for (int i = 0; i < VPL; i += 4) {
            float4 v = float4(*(device const vec<T, 4>*)(x + k + i));
            xsum += v.x + v.y + v.z + v.w;
            constexpr float s0 = 1.0f, s1 = 1.0f / 16, s2 = 1.0f / 256, s3 = 1.0f / 4096;
            const float base = (i % 8 == 0) ? 1.0f : 1.0f / 65536;  // second half of the word: nibbles 4..7
            xs[i] = v.x * (s0 * base), xs[i + 1] = v.y * (s1 * base), xs[i + 2] = v.z * (s2 * base),
            xs[i + 3] = v.w * (s3 * base);
        }
        const int g = k / GS;
        for (int r = 0; r < ROWS; ++r) {
            uint32_t q[WORDS];
            Words<WORDS>::load(wr[r] + k / 8, q);
            float dot = 0.0f;
            for (int j = 0; j < WORDS; ++j) {
                const thread float* xw = xs + 8 * j;
                dot += xw[0] * float(q[j] & 0x0000000Fu) + xw[1] * float(q[j] & 0x000000F0u) +
                       xw[2] * float(q[j] & 0x00000F00u) + xw[3] * float(q[j] & 0x0000F000u) +
                       xw[4] * float(q[j] & 0x000F0000u) + xw[5] * float(q[j] & 0x00F00000u) +
                       xw[6] * float(q[j] & 0x0F000000u) + xw[7] * float(q[j] & 0xF0000000u);
            }
            acc[r] += float(sr[r][g]) * dot + float(br[r][g]) * xsum;
        }
    }

    // Reduce over the LPR lanes of the row group (lane t == 0 ends up with the sum).
    for (int r = 0; r < ROWS; ++r)
        for (int off = LPR / 2; off > 0; off /= 2) acc[r] += simd_shuffle_down(acc[r], off);

    if (SPLIT == 1) {
        if (t == 0)
            for (int r = 0; r < ROWS; ++r)
                if (row0 + r < N) y[row0 + r] = T(acc[r]);
        return;
    }
    // Split-K: every slice parks its partial sums, slice 0 adds them up.
    threadgroup float (*partial)[ROWS_PER_SG] = (threadgroup float (*)[ROWS_PER_SG])partial_;
    if (t == 0)
        for (int r = 0; r < ROWS; ++r) partial[sg][(lane / LPR) * ROWS + r] = acc[r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0 && t == 0)
        for (int r = 0; r < ROWS; ++r) {
            float sum = 0.0f;
            for (int s = 0; s < SPLIT; ++s) sum += partial[sg + s][(lane / LPR) * ROWS + r];
            if (row0 + r < N) y[row0 + r] = T(sum);
        }
}

template <typename T, int GS, int LPR, int ROWS, int VPL, int SPLIT, int NSG>
kernel void qmv4(device const uint32_t* w, device const T* sc, device const T* bi, device const T* x, device T* y,
                 constant int& N, constant int& K, uint tg [[threadgroup_position_in_grid]],
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[NSG * (32 / LPR) * ROWS];
    qmv4_body<T, GS, LPR, ROWS, VPL, SPLIT, NSG>(w, sc, bi, x, y, N, K, tg, sg, lane, partial);
}

// Up to three independent matvecs (own x, own weight, same K) in one dispatch: threadgroups [0, T0) do the first,
// [T0, T0 + T1) the second, the rest the third. Saves the fixed latency of two small dispatches.
template <typename T, int GS, int LPR, int ROWS, int VPL, int SPLIT, int NSG>
kernel void qmv4_multi(device const uint32_t* w0, device const T* sc0, device const T* bi0, device const T* x0,
                       device T* y0, device const uint32_t* w1, device const T* sc1, device const T* bi1,
                       device const T* x1, device T* y1, device const uint32_t* w2, device const T* sc2,
                       device const T* bi2, device const T* x2, device T* y2, constant int& N0, constant int& N1,
                       constant int& N2, constant int& T0, constant int& T1, constant int& K,
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[NSG * (32 / LPR) * ROWS];
    if (int(tg) < T0) {
        qmv4_body<T, GS, LPR, ROWS, VPL, SPLIT, NSG>(w0, sc0, bi0, x0, y0, N0, K, tg, sg, lane, partial);
    } else if (int(tg) < T0 + T1) {
        qmv4_body<T, GS, LPR, ROWS, VPL, SPLIT, NSG>(w1, sc1, bi1, x1, y1, N1, K, tg - T0, sg, lane, partial);
    } else {
        qmv4_body<T, GS, LPR, ROWS, VPL, SPLIT, NSG>(w2, sc2, bi2, x2, y2, N2, K, tg - T0 - T1, sg, lane, partial);
    }
}

// Instantiations: qmv4_<T>_<GS>_<LPR>_<ROWS>_<VPL>_<SPLIT>_<NSG>, for the layouts qmv4.py's dispatcher picks (tune.py
// replaces everything after the @@INSTANTIATIONS@@ marker with its own sweep).
#define QMV4(T, GS, L, R, V, S, G)                                                                                  \
    template [[host_name("qmv4_" #T "_" #GS "_" #L "_" #R "_" #V "_" #S "_" #G)]] kernel void                       \
    qmv4<T, GS, L, R, V, S, G>(device const uint32_t*, device const T*, device const T*, device const T*, device T*, \
                               constant int&, constant int&, uint, uint, uint);
#define INSTANTIATE_ALL(L, R, V, S, G)                                                                               \
    QMV4(float, 32, L, R, V, S, G) QMV4(half, 32, L, R, V, S, G) QMV4(bfloat, 32, L, R, V, S, G)                    \
    QMV4(float, 64, L, R, V, S, G) QMV4(half, 64, L, R, V, S, G) QMV4(bfloat, 64, L, R, V, S, G)
#define QMV4_MULTI(T, GS, L, R, V, S, G)                                                                            \
    template [[host_name("qmv4_multi_" #T "_" #GS "_" #L "_" #R "_" #V "_" #S "_" #G)]] kernel void                 \
    qmv4_multi<T, GS, L, R, V, S, G>(device const uint32_t*, device const T*, device const T*, device const T*,       \
        device T*, device const uint32_t*, device const T*, device const T*, device const T*, device T*,             \
        device const uint32_t*, device const T*, device const T*, device const T*, device T*, constant int&,         \
        constant int&, constant int&, constant int&, constant int&, constant int&, uint, uint, uint);
QMV4_MULTI(bfloat, 64, 32, 2, 16, 1, 2) QMV4_MULTI(half, 64, 32, 2, 16, 1, 2) QMV4_MULTI(float, 64, 32, 2, 16, 1, 2)
// @@INSTANTIATIONS@@
INSTANTIATE_ALL(16, 1, 16, 8, 8)  // thin: N <= 512
INSTANTIATE_ALL(8, 1, 8, 1, 4)    // short K, tall
INSTANTIATE_ALL(8, 1, 16, 2, 4)   // short K
INSTANTIATE_ALL(32, 2, 16, 1, 2)  // tall
INSTANTIATE_ALL(8, 1, 16, 2, 2)   // mid
