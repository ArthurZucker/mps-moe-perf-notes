// Single-query (decode) attention for Inkling-2 on Apple GPUs.
//
// Semantics, per query head h (GQA: kv head = h / G), over the L cache slots j:
//
//     dist_j  = key_distance[j]                                   (query position - key position, < 0 = empty slot)
//     bias_j  = dist_j < R ? rel_states[h] . rel_proj_t[dist_j] : 0   (fused relative-position-bias projection)
//     s_j     = scaling * q[h] . k[j] + bias_j                     (slots with dist_j < 0 are masked out)
//     p       = softmax([s_0 .. s_{L-1}, sink[h]])                 (the sink column only absorbs probability mass)
//     out[h]  = sum_j p_j v[j]
//
// Flash-decoding (split-K) layout in a single dispatch:
//   - One threadgroup per (kv head, chunk of CHUNK slots). It reads its K/V slice once for all G query heads of the
//     kv head. Inside a simdgroup, LPK lanes cooperate on one key (128 / LPK dims each), so a simdgroup walks
//     32 / LPK adjacent keys at a time with fully coalesced row reads. Every key group keeps its own online-softmax
//     state; groups are merged with shuffles, then across simdgroups through threadgroup memory, and one
//     (max, sum, acc[128]) partial per query head and chunk is written to `part_o` / `part_ml`.
//     LPK = 16 (8 dims, one 16-byte K and V load per lane and key) is the sweet spot on an M3 Max: LPK = 8 needs
//     ~150 registers per thread (G = 3 queries + accumulators of 16 dims) and caps residency below what it takes to
//     keep enough loads in flight; LPK = 32 spends too much time in shuffles.
//   - The threadgroup that finishes last for a kv head (an atomic ticket per kv head in `counters`) merges the chunk
//     partials with the sink and writes the G output rows, then resets the ticket to 0 for the next call. This saves
//     the second dispatch of the classic two-kernel flash-decoding (one fewer launch per layer).
//
// Scores live in the log2 domain (q, rel_states and the sink are pre-multiplied by log2(e)) so every exponential is a
// native exp2. A finite sentinel replaces -inf because the shader is compiled with fast math.

#include <metal_stdlib>
using namespace metal;

constant constexpr int D = 128;           // head dim
constant constexpr int DREL = 16;         // relative-state dim (d_rel)
constant constexpr float LOG2E = 1.4426950408889634f;
constant constexpr float NEG_BIG = -1.0e30f;  // "minus infinity" that survives fast math
constant constexpr int MAX_CHUNKS = 64;       // upper bound on n_chunks (the host picks CHUNK >= L / 64)

template <int N, typename T>
inline void load_row(device const T* p, thread float* out) {
    // N consecutive elements as 4-wide vector loads.
    for (int i = 0; i < N; i += 4) {
        const float4 v = float4(*(device const vec<T, 4>*)(p + i));
        out[i] = v.x, out[i + 1] = v.y, out[i + 2] = v.z, out[i + 3] = v.w;
    }
}

template <typename T, typename IdxT, int G, int NSG, int LPK>
kernel void decode_attn(device const T* q [[buffer(0)]],                // [Hkv * G, D]
                        device const T* k [[buffer(1)]],                // [Hkv, L, D]
                        device const T* v [[buffer(2)]],                // [Hkv, L, D]
                        device const T* rel_states [[buffer(3)]],       // [Hkv * G, DREL]
                        device const T* rel_proj_t [[buffer(4)]],       // [R, DREL]
                        device const IdxT* key_distance [[buffer(5)]],  // [L]
                        device const T* sinks [[buffer(6)]],            // [Hkv * G]
                        device T* out [[buffer(7)]],                    // [Hkv * G, D]
                        device float* part_o [[buffer(8)]],             // [Hkv * G, n_chunks, D]   (workspace)
                        device float* part_ml [[buffer(9)]],            // [Hkv * G, n_chunks, 2]   (workspace)
                        device atomic_uint* counters [[buffer(10)]],    // [Hkv], zero between calls
                        constant int& L [[buffer(11)]], constant int& R [[buffer(12)]],
                        constant int& CHUNK [[buffer(13)]], constant float& scaling [[buffer(14)]],
                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int DPL = D / LPK;                       // head dims per lane
    constexpr int KPS = 32 / LPK;                      // keys per simdgroup step
    constexpr int RPL = DREL >= LPK ? DREL / LPK : 1;  // relative-state dims per lane (lanes past DREL hold none)
    constexpr int NG = NSG * KPS;                      // key groups per threadgroup
    constexpr int NT = NSG * 32;   // threads per threadgroup
    const int n_chunks = (L + CHUNK - 1) / CHUNK;
    const int kvh = tg / n_chunks, chunk = tg % n_chunks;
    const int kl = lane % LPK, grp = sg * KPS + lane / LPK;
    const int start = chunk * CHUNK, end = min(start + CHUNK, L);

    // This lane's slice of the G queries (scaled into the log2 domain) and of their relative states.
    float qf[G][DPL], rs[G][RPL];
    for (int g = 0; g < G; ++g) {
        const int h = kvh * G + g;
        load_row<DPL>(q + h * D + kl * DPL, qf[g]);
        for (int i = 0; i < DPL; ++i) qf[g][i] *= scaling * LOG2E;
        for (int r = 0; r < RPL; ++r)
            rs[g][r] = kl * RPL < DREL ? float(rel_states[h * DREL + kl * RPL + r]) * LOG2E : 0.0f;
    }

    float m[G], l[G], acc[G][DPL];
    for (int g = 0; g < G; ++g) {
        m[g] = NEG_BIG, l[g] = 0.0f;
        for (int i = 0; i < DPL; ++i) acc[g][i] = 0.0f;
    }

    // ---- partial pass over this chunk's slots ------------------------------------------------------------------
    // Skip chunks with no valid slot at all (the unfilled tail of a static cache): one read of the distances.
    threadgroup bool tg_any[NSG];
    bool any = false;
    for (int j = start + tid; j < end; j += NT) any = any || key_distance[j] >= 0;
    any = simd_any(any);
    if (lane == 0) tg_any[sg] = any;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    any = false;
    for (int s = 0; s < NSG; ++s) any = any || tg_any[s];

    // K/V rows are loaded unconditionally (every slot index is in bounds) and empty slots are masked with selects,
    // so no load waits on the distance load and the loop stays branch-free.
    device const T* kb = k + (long)kvh * L * D + kl * DPL;
    device const T* vb = v + (long)kvh * L * D + kl * DPL;
    for (int j = start + grp; any && j < end; j += NG) {
        float kf[DPL], vf[DPL];
        load_row<DPL>(kb + (long)j * D, kf);
        load_row<DPL>(vb + (long)j * D, vf);
        const IdxT dist = key_distance[j];
        const bool valid = dist >= 0;
        for (int i = 0; i < DPL; ++i) vf[i] = valid ? vf[i] : 0.0f;  // an empty slot may hold garbage (NaN * 0)

        // Fused relative bias: this lane's RPL terms of rel_states[h] . rel_proj_t[dist]; the LPK-lane sum below adds
        // them to q.k. Zero beyond rel_extent (and for empty slots, which are masked anyway).
        float pr[RPL];
        const float in_range = (valid && dist < R) ? 1.0f : 0.0f;
        const long prow = (long)clamp(dist, IdxT(0), IdxT(R - 1)) * DREL + min(kl * RPL, DREL - RPL);
        for (int r = 0; r < RPL; ++r) pr[r] = float(rel_proj_t[prow + r]) * in_range;

        float s[G];
        for (int g = 0; g < G; ++g) {
            float dot = 0.0f;
            for (int i = 0; i < DPL; ++i) dot = fma(qf[g][i], kf[i], dot);
            for (int r = 0; r < RPL; ++r) dot = fma(rs[g][r], pr[r], dot);
            s[g] = dot;
        }
        for (int g = 0; g < G; ++g)
            for (ushort off = 1; off < LPK; off *= 2) s[g] += simd_shuffle_xor(s[g], off);

        for (int g = 0; g < G; ++g) {
            const float mn = valid ? max(m[g], s[g]) : m[g];
            const float a = exp2(m[g] - mn), p = valid ? exp2(s[g] - mn) : 0.0f;
            l[g] = fma(l[g], a, p);
            for (int i = 0; i < DPL; ++i) acc[g][i] = fma(p, vf[i], acc[g][i] * a);
            m[g] = mn;
        }
    }

    // Merge the KPS key groups of this simdgroup (lanes kl, kl + LPK, ...).
    for (ushort off = LPK; off < 32; off *= 2) {
        for (int g = 0; g < G; ++g) {
            const float mo = simd_shuffle_xor(m[g], off), lo = simd_shuffle_xor(l[g], off);
            const float mn = max(m[g], mo);
            const float a = exp2(m[g] - mn), b = exp2(mo - mn);
            l[g] = l[g] * a + lo * b;
            for (int i = 0; i < DPL; ++i) acc[g][i] = acc[g][i] * a + simd_shuffle_xor(acc[g][i], off) * b;
            m[g] = mn;
        }
    }

    // Merge across simdgroups through threadgroup memory and write this chunk's partial.
    threadgroup float tg_acc[NSG][G][D];
    threadgroup float tg_m[NSG][G], tg_l[NSG][G];
    if (lane < LPK)
        for (int g = 0; g < G; ++g) {
            for (int i = 0; i < DPL; ++i) tg_acc[sg][g][kl * DPL + i] = acc[g][i];
            if (lane == 0) tg_m[sg][g] = m[g], tg_l[sg][g] = l[g];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < G * D; idx += NT) {
        const int g = idx / D, d = idx % D;
        float mx = NEG_BIG;
        for (int s = 0; s < NSG; ++s) mx = max(mx, tg_m[s][g]);
        float o = 0.0f, sum = 0.0f;
        for (int s = 0; s < NSG; ++s) {
            const float w = exp2(tg_m[s][g] - mx);
            o = fma(tg_acc[s][g][d], w, o);
            sum = fma(tg_l[s][g], w, sum);
        }
        const long row = (long)(kvh * G + g) * n_chunks + chunk;
        part_o[row * D + d] = o;
        if (d == 0) part_ml[row * 2] = mx, part_ml[row * 2 + 1] = sum;
    }

    // ---- the last threadgroup of this kv head merges the n_chunks partials -------------------------------------
    // Release this threadgroup's partial device-wide before taking a ticket; the threadgroup that draws the last
    // ticket acquires every other chunk's partial.
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup bool is_last;
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        is_last = atomic_fetch_add_explicit(&counters[kvh], 1u, memory_order_relaxed) == uint(n_chunks - 1);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (!is_last) return;
    if (tid == 0) atomic_store_explicit(&counters[kvh], 0u, memory_order_relaxed);  // ready for the next call

    // Per-(head, chunk) merge weights exp2(m_c - M) / denominator, computed once into threadgroup memory (all loads
    // issued in parallel), then every output element is an independent weighted sum over the chunks.
    threadgroup float tg_w[G][MAX_CHUNKS];
    threadgroup float tg_inv[G], tg_mx[G];
    for (int idx = tid; idx < G * n_chunks; idx += NT) {
        const int g = idx / n_chunks, c = idx % n_chunks;
        tg_w[g][c] = part_ml[((long)(kvh * G + g) * n_chunks + c) * 2];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < G) {
        const int h = kvh * G + tid;
        const float sink = float(sinks[h]) * LOG2E;
        float mx = sink;
        for (int c = 0; c < n_chunks; ++c) mx = max(mx, tg_w[tid][c]);
        float sum = exp2(sink - mx);
        for (int c = 0; c < n_chunks; ++c) sum = fma(part_ml[((long)h * n_chunks + c) * 2 + 1], exp2(tg_w[tid][c] - mx), sum);
        tg_mx[tid] = mx, tg_inv[tid] = 1.0f / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < G * n_chunks; idx += NT) {
        const int g = idx / n_chunks, c = idx % n_chunks;
        tg_w[g][c] = exp2(tg_w[g][c] - tg_mx[g]) * tg_inv[g];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < G * D; idx += NT) {
        const int g = idx / D, d = idx % D, h = kvh * G + g;
        device const float* po = part_o + (long)h * n_chunks * D + d;
        float o = 0.0f;
        for (int c = 0; c < n_chunks; ++c) o = fma(po[(long)c * D], tg_w[g][c], o);
        out[h * D + d] = T(o);
    }
}

#define DECODE_ATTN(T, I, G, NSG, LPK)                                                                             \
    template [[host_name("decode_attn_" #T "_" #I "_" #G "_" #NSG "_" #LPK)]] kernel void                        \
    decode_attn<T, I, G, NSG, LPK>(                                                                               \
        device const T*, device const T*, device const T*, device const T*, device const T*, device const I*,     \
        device const T*, device T*, device float*, device float*, device atomic_uint*, constant int&,             \
        constant int&, constant int&, constant float&, uint, uint, uint, uint);
// LPK = 16 (see the header). Threadgroup memory holds NSG * G * 128 floats, so the widest threadgroups are only
// instantiated for small G; the host keeps NSG * G <= 48.
#define DECODE_ATTN_SMALL_G(T, I, G) DECODE_ATTN(T, I, G, 4, 16) DECODE_ATTN(T, I, G, 8, 16) DECODE_ATTN(T, I, G, 16, 16)
#define DECODE_ATTN_LARGE_G(T, I, G) DECODE_ATTN(T, I, G, 2, 16) DECODE_ATTN(T, I, G, 4, 16) DECODE_ATTN(T, I, G, 8, 16)
#define DECODE_ATTN_I(T, I)                                                                                     \
    DECODE_ATTN_SMALL_G(T, I, 1) DECODE_ATTN_SMALL_G(T, I, 2) DECODE_ATTN_SMALL_G(T, I, 3) DECODE_ATTN_LARGE_G(T, I, 4) \
    DECODE_ATTN_LARGE_G(T, I, 6)
#define DECODE_ATTN_T(T) DECODE_ATTN_I(T, int) DECODE_ATTN_I(T, long)
DECODE_ATTN_T(float)
DECODE_ATTN_T(half)
DECODE_ATTN_T(bfloat)
