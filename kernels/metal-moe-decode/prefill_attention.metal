// Prefill (T > 1) attention for Inkling-2 on Apple GPUs: flash attention with the hidden-state conditioned relative
// position bias, the sliding window / causal mask and the per-head sinks computed on the fly (nothing [H, T, L] is
// ever materialized). Semantics, query i (position base + i over key positions j), head h, kv head h / G:
//     dist  = base + i - j;  valid = 0 <= dist < window and j < L
//     bias  = dist < R ? rel_states[i, h] . proj_t[dist] : 0
//     s_j   = scaling * q[i, h] . k[j] + bias            p = softmax([s_valid..., sink[i, h]])      out = sum_j p_j v[j]
// Threadgroup = (head, block of 32 queries), 4 simdgroups x 8 query rows. Q / K / V fragments come straight from device
// memory into bf16 8x8 simdgroup matrices (fp32 accumulation); scores and bf16 probabilities pass through threadgroup
// memory; online softmax in the log2 domain; the key loop only visits the blocks of the (band-shaped) valid range.
#include <metal_stdlib>
using namespace metal;
constant constexpr int D = 128, DREL = 16, BQ = 32, BK = 32;
constant constexpr float LOG2E = 1.4426950408889634f, NEG_BIG = -1.0e30f;

template <typename T>
kernel void prefill_attention(device const T* q,            // [H, Tp, D]
                              device const T* k,            // [Hkv, L, D]
                              device const T* v,            // [Hkv, L, D]
                              device const float* rs,       // [Tp, H, DREL]
                              device const float* proj_t,   // [R, DREL]
                              device const float* sinks,    // [Tp, H]
                              device const long* distance,  // [T, L]: only [0, 0] (= base) is read
                              device T* out,                // [Tp, H, D]
                              constant int& H, constant int& G, constant int& T_, constant int& Tp, constant int& L,
                              constant int& R, constant int& window, constant float& scaling,
                              uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float s_tile[4][8 * BK];
    threadgroup bfloat p_tile[4][8 * BK];
    threadgroup float diag[4][64];
    threadgroup float o_tile[4][8 * D];
    const int h = tg.x, kvh = h / G, i0 = tg.y * BQ, r0 = i0 + int(sg) * 8;
    const long base = distance[0];
    device const T* qh = q + ((long)h * Tp + r0) * D;
    device const T* kh = k + (long)kvh * L * D;
    device const T* vh = v + (long)kvh * L * D;

    simdgroup_matrix<T, 8, 8> qf[D / 8];
    for (int d = 0; d < D / 8; ++d) simdgroup_load(qf[d], qh + d * 8, D);
    simdgroup_float8x8 of[D / 8];
    for (int d = 0; d < D / 8; ++d) of[d] = simdgroup_float8x8(0.0f);

    // this lane's row (r) and 8 columns (c0..c0+7) of the 8 x 32 score tile
    const int r = lane / 4, c0 = (lane % 4) * 8, i = r0 + r;
    float rsv[DREL];
    for (int e = 0; e < DREL; ++e) rsv[e] = rs[((long)min(i, Tp - 1) * H + h) * DREL + e];
    float m = NEG_BIG, l = 0.0f;

    const int i_last = min(i0 + BQ - 1, T_ - 1);
    const long j_lo = max(0L, base + i0 - (long)(window - 1)), j_hi = min((long)L - 1, base + i_last);
    for (long j0 = (j_lo / BK) * BK; j0 <= j_hi; j0 += BK) {
        // S = Q K^T for this simdgroup's 8 rows x 32 keys
        simdgroup_float8x8 sf[BK / 8];
        for (int n = 0; n < BK / 8; ++n) sf[n] = simdgroup_float8x8(0.0f);
        for (int d = 0; d < D / 8; ++d) {
            for (int n = 0; n < BK / 8; ++n) {
                simdgroup_matrix<T, 8, 8> kt;
                simdgroup_load(kt, kh + (j0 + 8 * n) * D + d * 8, D, 0, true);
                simdgroup_multiply_accumulate(sf[n], qf[d], kt, sf[n]);
            }
        }
        for (int n = 0; n < BK / 8; ++n) simdgroup_store(sf[n], &s_tile[sg][n * 8], BK);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float sv[8], mloc = NEG_BIG;
        for (int c = 0; c < 8; ++c) {
            const long j = j0 + c0 + c, dist = base + i - j;
            const bool valid = i < T_ && j < L && dist >= 0 && dist < window;
            float bias = 0.0f;
            if (valid && dist < R) {
                device const float* pr = proj_t + dist * DREL;
                for (int e = 0; e < DREL; ++e) bias += rsv[e] * pr[e];
            }
            sv[c] = valid ? (s_tile[sg][r * BK + c0 + c] * scaling + bias) * LOG2E : NEG_BIG;
            mloc = max(mloc, sv[c]);
        }
        mloc = max(mloc, simd_shuffle_xor(mloc, 1));
        mloc = max(mloc, simd_shuffle_xor(mloc, 2));
        const float m_new = max(m, mloc), alpha = exp2(m - m_new);
        float lsum = 0.0f;
        for (int c = 0; c < 8; ++c) {
            const float p = sv[c] > NEG_BIG * 0.5f ? exp2(sv[c] - m_new) : 0.0f;
            lsum += p;
            p_tile[sg][r * BK + c0 + c] = bfloat(p);
        }
        lsum += simd_shuffle_xor(lsum, 1);
        lsum += simd_shuffle_xor(lsum, 2);
        l = l * alpha + lsum;
        m = m_new;
        // O <- diag(alpha) O, then O += P V
        if (lane < 8) for (int c = 0; c < 8; ++c) diag[sg][lane * 8 + c] = 0.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane % 4 == 0) diag[sg][r * 8 + r] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dg;
        simdgroup_load(dg, diag[sg], 8);
        for (int d = 0; d < D / 8; ++d) simdgroup_multiply(of[d], dg, of[d]);
        for (int kb = 0; kb < BK / 8; ++kb) {
            simdgroup_matrix<T, 8, 8> pf;
            simdgroup_load(pf, (threadgroup const T*)&p_tile[sg][kb * 8], BK);
            for (int d = 0; d < D / 8; ++d) {
                simdgroup_matrix<T, 8, 8> vf;
                simdgroup_load(vf, vh + (j0 + 8 * kb) * D + d * 8, D);
                simdgroup_multiply_accumulate(of[d], pf, vf, of[d]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // sink column, normalization, store
    const float sink = sinks[(long)min(i, Tp - 1) * H + h] * LOG2E;
    const float m_f = max(m, sink), total = l * exp2(m - m_f) + exp2(sink - m_f);
    const float factor = exp2(m - m_f) / total;
    if (lane < 8) for (int c = 0; c < 8; ++c) diag[sg][lane * 8 + c] = 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane % 4 == 0) diag[sg][r * 8 + r] = factor;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dg;
    simdgroup_load(dg, diag[sg], 8);
    for (int d = 0; d < D / 8; ++d) {
        simdgroup_multiply(of[d], dg, of[d]);
        simdgroup_store(of[d], &o_tile[sg][d * 8], D);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = lane; e < 8 * D; e += 32) {
        const int rr = e / D, dd = e % D;
        out[((long)(r0 + rr) * H + h) * D + dd] = T(o_tile[sg][e]);
    }
}
#define PA(T) template [[host_name("prefill_attention_" #T)]] kernel void prefill_attention<T>(device const T*, \
    device const T*, device const T*, device const float*, device const float*, device const float*, device const long*, \
    device T*, constant int&, constant int&, constant int&, constant int&, constant int&, constant int&, constant int&, \
    constant float&, uint2, uint, uint);
PA(bfloat) PA(half)
