// Metal kernels for the linear-attention layers of hybrid models (Qwen3-Next style gated delta nets, short convs).
//
//   causal_conv1d_fwd        depthwise causal conv over a whole sequence (prefill), any input strides
//   causal_conv1d_step       one decode token: conv over [state, x], state shifted in place
//   gated_delta_rule         token-by-token gated delta rule (decode, short sequences), state advanced in place
//   gated_delta_rule_chunked same, for long sequences: q/k/v staged through threadgroup memory 16 tokens at a time
//   gated_delta_net_decode   one decode token of a whole gated delta net core: conv step + q/k/v split + q/k l2
//                            norm + gates (g, beta) + delta rule, in one dispatch
//
// Delta rule layout. The state of a (batch, value head) is a [DK, DV] matrix; each of its columns evolves
// independently given q_t, k_t (shared by the column) and v_t[col]:
//   S = S * exp(g_t);   S += k_t (x) (v_t - S^T k_t) * beta_t;   o_t = S^T q_t / sqrt(DK)
// so the column space is split freely. A thread owns 4 adjacent columns (vector loads/stores of the state) and
// R = DK / RS rows of them; the RS threads sharing a column group are adjacent lanes of one simdgroup, so the dot
// products over DK reduce with RS-wide simd shuffles: no threadgroup memory, no barriers. A simdgroup covers
// 4 * (32 / RS) columns; the grid is (simdgroups over all heads' column blocks, batch).

#include <metal_stdlib>
using namespace metal;

inline float silu_if(float x, int silu) { return silu ? x / (1.0f + exp(-x)) : x; }

template <typename T> inline float4 ld4(device const T* p) { return float4(*(device const vec<T, 4>*)p); }
template <typename T> inline void st4(device T* p, float4 x) { *(device vec<T, 4>*)p = vec<T, 4>(x); }

// ------------------------------------------------------------------------------------------------ causal conv1d

// out[b, c, t] = act(sum_j w[c, j] * x[b, c, t - K + 1 + j] + bias[c]), zero padded on the left. x and out are
// addressed through (batch, channel, time) strides, so a channels-last [B, L, C] buffer (the in_proj output viewed as
// [B, C, L]) is read and written without a transpose copy. `c_fast`: grid x runs over channels (channels-last
// layouts, coalesced) instead of time.
template <typename T>
kernel void causal_conv1d_fwd(device const T* x, device const T* w, device const T* bias, device T* out,
                              constant int& C, constant int& L, constant int& K, constant int& has_bias,
                              constant int& silu, constant int& xb, constant int& xc, constant int& xt,
                              constant int& ob, constant int& oc, constant int& ot, constant int& c_fast,
                              uint3 tid [[thread_position_in_grid]]) {
    int c = c_fast ? tid.x : tid.y, t = c_fast ? tid.y : tid.x, b = tid.z;
    if (c >= C || t >= L) return;
    device const T* row = x + b * xb + c * xc;
    float acc = has_bias ? float(bias[c]) : 0.0f;
    for (int j = 0; j < K; ++j) {
        int s = t - K + 1 + j;
        if (s >= 0) acc += float(w[c * K + j]) * float(row[s * xt]);
    }
    out[b * ob + c * oc + t * ot] = T(silu_if(acc, silu));
}

// One new token: window = last K of [state, x]; state [B, C, N] shifts left by one and takes x.
template <typename T, typename S>
kernel void causal_conv1d_step(device const T* x, device S* state, device const T* w, device const T* bias,
                               device T* out, constant int& C, constant int& N, constant int& K,
                               constant int& has_bias, constant int& silu, uint2 tid [[thread_position_in_grid]]) {
    int c = tid.x, b = tid.y;
    if (c >= C) return;
    device S* st = state + (b * C + c) * N;
    float xt = float(x[b * C + c]);
    float acc = has_bias ? float(bias[c]) : 0.0f;
    for (int j = 0; j < K - 1; ++j) acc += float(w[c * K + j]) * float(st[N - K + 1 + j]);
    acc += float(w[c * K + K - 1]) * xt;
    for (int j = 0; j < N - 1; ++j) st[j] = st[j + 1];
    st[N - 1] = S(xt);
    out[b * C + c] = T(silu_if(acc, silu));
}

// ------------------------------------------------------------------------------------------- gated delta rule

// One token of the delta rule on a thread's [R rows x 4 cols] tile, from raw (un-normalized) q/k rows. Everything
// that needs a sum over DK goes through ONE shuffle pass: S^T k, S^T q (old state), |q|^2, |k|^2 and k.q, since
//   S' = decay S + k_n (x) d,  d = (v - decay S^T k_n) beta,  o = S'^T q_n = decay S^T q_n + (k_n . q_n) d
// (k_n, q_n normalized, q_n also scaled by 1/sqrt(DK)). Returns o for the 4 columns on every lane of the group.
template <int R, int RS>
inline float4 delta_step(thread float4 (&s)[R], thread const float (&kr)[R], thread const float (&qr)[R],
                         float4 vv, float decay, float beta, int l2norm, int DK) {
    float4 a = 0.0f, c = 0.0f;
    float3 n = 0.0f;  // |q|^2, |k|^2, k.q over this thread's rows
#pragma unroll
    for (int r = 0; r < R; ++r) {
        a += s[r] * kr[r];
        c += s[r] * qr[r];
        n += float3(qr[r] * qr[r], kr[r] * kr[r], kr[r] * qr[r]);
    }
#pragma unroll
    for (ushort m = 1; m < RS; m <<= 1) {
        a += simd_shuffle_xor(a, m);
        c += simd_shuffle_xor(c, m);
        n += simd_shuffle_xor(n, m);
    }
    float nq = (l2norm ? rsqrt(n.x + 1e-6f) : 1.0f) * rsqrt(float(DK));
    float nk = l2norm ? rsqrt(n.y + 1e-6f) : 1.0f;
    float4 d = (vv - a * (decay * nk)) * beta;
    float4 dk = d * nk;
#pragma unroll
    for (int r = 0; r < R; ++r) s[r] = s[r] * decay + kr[r] * dk;
    return (c * decay + d * (n.z * nk)) * nq;
}

template <typename T, int R>
inline void load_rows(device const T* p, thread float (&x)[R]) {
#pragma unroll
    for (int r = 0; r < R; r += 4) {
        float4 y = ld4(p + r);
        x[r] = y.x; x[r + 1] = y.y; x[r + 2] = y.z; x[r + 3] = y.w;
    }
}

// q, k: [B, L, HK, DK], v: [B, L, HV, DV], each with heads/dims contiguous and (batch, token) strides in elements;
// g (float, log decay), beta: [B, L, HV] contiguous; state [B, HV, DK, DV] advanced in place; out [B, L, HV, DV].
// Value head h reads key head h / (HV / HK). The next token's inputs are loaded before the current token's math so
// their latency hides behind it (the recurrence is latency-bound in prefill).
template <typename T, typename S, int DK, int RS>
kernel void gated_delta_rule(device const T* q, device const T* k, device const T* v, device const float* g,
                             device const T* beta, device S* state, device T* out,
                             constant int& L, constant int& HK, constant int& HV, constant int& DV,
                             constant int& qb, constant int& qt, constant int& kb, constant int& kt,
                             constant int& vb, constant int& vt, constant int& l2norm,
                             uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint nsg [[simdgroups_per_threadgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
    constexpr int R = DK / RS, CG = 32 / RS;
    const int rs = lane % RS, cg = lane / RS;
    const int blocks = DV / (4 * CG), blk = tg.x * nsg + sg, h = blk / blocks, b = tg.y;
    if (h >= HV) return;
    const int col = (blk % blocks) * 4 * CG + cg * 4, hk = h / (HV / HK);

    device S* st = state + (ulong(b * HV + h) * DK + rs * R) * DV + col;
    float4 s[R];
#pragma unroll
    for (int r = 0; r < R; ++r) s[r] = ld4(st + r * DV);

    device const T* qp = q + b * qb + hk * DK + rs * R;
    device const T* kp = k + b * kb + hk * DK + rs * R;
    device const T* vp = v + b * vb + h * DV + col;
    device const float* gp = g + b * L * HV + h;
    device const T* bp = beta + b * L * HV + h;
    device T* op = out + (ulong(b) * L * HV + h) * DV + col;

    float qn[R], kn[R];  // next token's inputs
    float4 vn = ld4(vp);
    float gn = gp[0], bn = float(bp[0]);
    load_rows<T, R>(qp, qn);
    load_rows<T, R>(kp, kn);
    for (int t = 0; t < L; ++t) {
        float qr[R], kr[R];
#pragma unroll
        for (int r = 0; r < R; ++r) { qr[r] = qn[r]; kr[r] = kn[r]; }
        float4 vv = vn;
        float decay = exp(gn), bt = bn;
        if (t + 1 < L) {
            vn = ld4(vp + (t + 1) * vt);
            gn = gp[(t + 1) * HV];
            bn = float(bp[(t + 1) * HV]);
            load_rows<T, R>(qp + (t + 1) * qt, qn);
            load_rows<T, R>(kp + (t + 1) * kt, kn);
        }
        float4 o = delta_step<R, RS>(s, kr, qr, vv, decay, bt, l2norm, DK);
        if (rs == 0) st4(op + ulong(t) * HV * DV, o);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) st4(st + r * DV, s[r]);
}

// ---------------------------------------------------------------------------- prefill: chunk-staged recurrence

// One token of the delta rule in the form the prefill kernel uses. The thread's tile holds s = S / D, where D is the
// running product of the decays (a scalar shared by the whole head), so the per-element decay multiply disappears:
//   D' = D * decay,  a = D' s^T k,  c = D' s^T q,  d = (v - a) * beta,  o = c + (k.q) d,  s += k (x) d / D'
// i.e. 3 FMAs per state element and token, and the two DK reductions (s^T k, s^T q) share one shuffle pass. kq is
// k.q of the normalized vectors. D is folded back into s before it can underflow (a uniform branch, all lanes agree).
template <int R, int RS>
inline float4 delta_step_lazy(thread float4 (&s)[R], thread const float (&kr)[R], thread const float (&qr)[R],
                              float4 vv, float kq, float decay, float beta, thread float& D) {
    if (D * decay < 1e-20f) {
#pragma unroll
        for (int r = 0; r < R; ++r) s[r] *= D;
        D = 1.0f;
    }
    D *= max(decay, 1e-30f);
    float4 a = 0.0f, c = 0.0f;
#pragma unroll
    for (int r = 0; r < R; ++r) { a += s[r] * kr[r]; c += s[r] * qr[r]; }
#pragma unroll
    for (ushort m = 1; m < RS; m <<= 1) { a += simd_shuffle_xor(a, m); c += simd_shuffle_xor(c, m); }
    float4 d = (vv - a * D) * beta;
    float4 dd = d * (1.0f / D);
#pragma unroll
    for (int r = 0; r < R; ++r) s[r] += kr[r] * dd;
    return c * D + kq * d;
}

// Same recurrence and contract as `gated_delta_rule`, for long sequences. The per-token math is short; fetching each
// token's q/k/v from DRAM is not. So a threadgroup (4 simdgroups: W columns of ONE value head) stages TC tokens at a
// time into threadgroup memory, paying the DRAM latency once per chunk (other threadgroups on the core run
// meanwhile; prefetching the next chunk into registers measured ~30% slower: the live registers cost occupancy, and
// this kernel is throughput-bound at 48 heads). q/k are l2-normalized (q also scaled by 1/sqrt(DK)) and k.q
// computed once while staging, instead of by every column group per token. Two barriers per chunk.
template <typename T, typename S, int DK, int RS>
kernel void gated_delta_rule_chunked(device const T* q, device const T* k, device const T* v, device const float* g,
                                     device const T* beta, device S* state, device T* out,
                                     constant int& L, constant int& HK, constant int& HV, constant int& DV,
                                     constant int& qb, constant int& qt, constant int& kb, constant int& kt,
                                     constant int& vb, constant int& vt, constant int& l2norm,
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint tid [[thread_index_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
    constexpr int R = DK / RS, CG = 32 / RS, NSG = 4, NT = 32 * NSG;
    constexpr int W = NSG * 4 * CG;                 // columns per threadgroup
    constexpr int TC = DK <= 128 ? 16 : 2048 / DK;  // tokens per chunk: q + k staged as float, <= 16 KB
    constexpr int PIECES = 2 * TC * DK / 8;         // 8-element pieces of the chunk's q rows, then its k rows
    constexpr int PER_T = PIECES / NT;              // per thread; piece m and m + PER_T / 2 are q and k of one spot
    constexpr int ROW = DK / 8;                     // adjacent lanes sharing one q/k row
    constexpr int VPIECES = TC * W / 4, PV = (VPIECES + NT - 1) / NT;  // 4-wide v pieces, per thread
    static_assert(PIECES % (2 * NT) == 0 && ROW <= 32 && R % 4 == 0, "staging layout");

    threadgroup float4 qs[TC][DK / 4], ks[TC][DK / 4], vs[TC][W / 4];
    threadgroup float gs[TC], bs[TC], kqs[TC];

    const int rs = lane % RS, cg = lane / RS;
    const int tgs_per_head = DV / W, h = tg.x / tgs_per_head, b = tg.y;
    const int col0 = (tg.x % tgs_per_head) * W, lc = sg * 4 * CG + cg * 4, hk = h / (HV / HK);

    device S* st = state + (ulong(b * HV + h) * DK + rs * R) * DV + col0 + lc;
    float4 s[R];
#pragma unroll
    for (int r = 0; r < R; ++r) s[r] = ld4(st + r * DV);
    float D = 1.0f;

    device const T* qp = q + b * qb + hk * DK;
    device const T* kp = k + b * kb + hk * DK;
    device const T* vp = v + b * vb + h * DV + col0;
    device const float* gp = g + b * L * HV + h;
    device const T* bp = beta + b * L * HV + h;
    device T* op = out + (ulong(b) * L * HV + h) * DV + col0 + lc;

    // One chunk on its way from DRAM to threadgroup memory, in the input dtype.
    vec<T, 4> pa[PER_T], pb[PER_T], pv[PV];
    float pg = 0.0f, pbeta = 0.0f;
    auto fetch = [&](int c0) {
#pragma unroll
        for (int m = 0; m < PER_T; ++m) {
            int piece = tid + m * NT, isk = m >= PER_T / 2, pp = piece - isk * (PIECES / 2);
            int t = c0 + pp / ROW, off = (pp % ROW) * 8;
            pa[m] = pb[m] = vec<T, 4>(0);
            if (t < L) {
                device const T* src = (isk ? kp + t * kt : qp + t * qt) + off;
                pa[m] = *(device const vec<T, 4>*)src;
                pb[m] = *(device const vec<T, 4>*)(src + 4);
            }
        }
#pragma unroll
        for (int m = 0; m < PV; ++m) {
            int piece = tid + m * NT, t = c0 + piece / (W / 4);
            pv[m] = piece < VPIECES && t < L ? *(device const vec<T, 4>*)(vp + t * vt + (piece % (W / 4)) * 4)
                                             : vec<T, 4>(0);
        }
        if (tid < TC && c0 + int(tid) < L) {
            pg = gp[(c0 + tid) * HV];
            pbeta = float(bp[(c0 + tid) * HV]);
        }
    };
    for (int c0 = 0; c0 < L; c0 += TC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);  // the previous chunk is no longer read
        fetch(c0);
#pragma unroll
        for (int m = 0; m < PER_T / 2; ++m) {
            // q piece m and k piece m + PER_T / 2 cover the same token and dims; a row's ROW lanes are adjacent
            float4 qa = float4(pa[m]), qb_ = float4(pb[m]);
            float4 ka = float4(pa[m + PER_T / 2]), kb_ = float4(pb[m + PER_T / 2]);
            float3 acc = float3(dot(qa, qa) + dot(qb_, qb_), dot(ka, ka) + dot(kb_, kb_), dot(qa, ka) + dot(qb_, kb_));
#pragma unroll
            for (ushort x = 1; x < ROW; x <<= 1) acc += simd_shuffle_xor(acc, x);
            float nq = (l2norm ? rsqrt(acc.x + 1e-6f) : 1.0f) * rsqrt(float(DK));
            float nk = l2norm ? rsqrt(acc.y + 1e-6f) : 1.0f;
            int pp = tid + m * NT, t = pp / ROW, i = (pp % ROW) * 2;
            qs[t][i] = qa * nq; qs[t][i + 1] = qb_ * nq;
            ks[t][i] = ka * nk; ks[t][i + 1] = kb_ * nk;
            if (pp % ROW == 0) kqs[t] = acc.z * nq * nk;
        }
#pragma unroll
        for (int m = 0; m < PV; ++m) {
            int piece = tid + m * NT;
            if (piece < VPIECES) vs[piece / (W / 4)][piece % (W / 4)] = float4(pv[m]);
        }
        if (tid < TC) { gs[tid] = exp(pg); bs[tid] = pbeta; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int n = min(TC, L - c0);
        for (int t = 0; t < n; ++t) {
            float kr[R], qr[R];
#pragma unroll
            for (int r = 0; r < R; r += 4) {
                float4 kk = ks[t][(rs * R + r) / 4], qq = qs[t][(rs * R + r) / 4];
                kr[r] = kk.x; kr[r + 1] = kk.y; kr[r + 2] = kk.z; kr[r + 3] = kk.w;
                qr[r] = qq.x; qr[r + 1] = qq.y; qr[r + 2] = qq.z; qr[r + 3] = qq.w;
            }
            float4 o = delta_step_lazy<R, RS>(s, kr, qr, vs[t][lc / 4], kqs[t], gs[t], bs[t], D);
            if (rs == 0) st4(op + ulong(c0 + t) * HV * DV, o);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) st4(st + r * DV, s[r] * D);
}

// ------------------------------------------------------------------------------------ fused decode (one token)

// conv over the window [state[N-K+1:], x] of channel c (conv state [N] per channel, weights [K] per channel).
template <typename T>
inline float conv_tap(device const T* cs, device const T* w, device const T* bias, float x, int c, int N, int K,
                      int has_bias, int silu) {
    float acc = has_bias ? float(bias[c]) : 0.0f;
    device const T* st = cs + c * N + N - K + 1;
    device const T* wc = w + c * K;
    for (int j = 0; j < K - 1; ++j) acc += float(wc[j]) * float(st[j]);
    return silu_if(acc + float(wc[K - 1]) * x, silu);
}

template <typename T>
inline void conv_shift(device T* cs, T x, int N) {
    for (int j = 0; j < N - 1; ++j) cs[j] = cs[j + 1];
    cs[N - 1] = x;
}

// One decode token of a gated delta net core:
//   qkv  = act(causal_conv1d_step(x, conv_state))                  x = in_proj_qkv output [B, 1, 2*HK*DK + HV*DV]
//   q, k, v = split(qkv);  q, k = l2norm(q), l2norm(k)  (if l2norm)
//   g    = -exp(A_log) * softplus(a + dt_bias);   beta = sigmoid(b)
//   out  = gated_delta_rule step on `state` (in place), conv_state shifted in place
// Same thread layout as `gated_delta_rule`. Each thread convolves the q/k channels of its rows and the v channels of
// its columns itself (the inputs are a few KB, cache resident). v channels have one reader, which shifts their conv
// state right away. q/k channels are read by every simdgroup of the key head (HV / HK value heads x column blocks),
// so the last of them to finish (an atomic arrival counter per (batch, key head), reset by that same simdgroup)
// shifts their state: no reader can still be using the old one.
template <typename T, typename S, int DK, int RS>
kernel void gated_delta_net_decode(device const T* x, device T* conv_state, device const T* conv_w,
                                   device const T* conv_b, device const T* a, device const T* bg,
                                   device const float* A_log, device const float* dt_bias, device S* state,
                                   device T* out, device atomic_int* arrivals,
                                   constant int& N, constant int& K, constant int& has_bias, constant int& silu,
                                   constant int& HK, constant int& HV, constant int& DV, constant int& xb,
                                   constant int& l2norm,
                                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                   uint nsg [[simdgroups_per_threadgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
    constexpr int R = DK / RS, CG = 32 / RS;
    const int rs = lane % RS, cg = lane / RS;
    const int blocks = DV / (4 * CG), blk = tg.x * nsg + sg, h = blk / blocks, b = tg.y;
    if (h >= HV) return;
    const int col = (blk % blocks) * 4 * CG + cg * 4, hk = h / (HV / HK);
    const int C = 2 * HK * DK + HV * DV;

    device const T* xr = x + b * xb;
    device T* cs = conv_state + ulong(b) * C * N;
    device S* st = state + (ulong(b * HV + h) * DK + rs * R) * DV + col;

    // Issue the state loads first: they are the only DRAM traffic, the rest hits cache.
    float4 s[R];
#pragma unroll
    for (int r = 0; r < R; ++r) s[r] = ld4(st + r * DV);

    // Conv + split: this thread's q/k rows and v columns.
    const int cq = hk * DK + rs * R, ck = HK * DK + cq, cv = 2 * HK * DK + h * DV + col;
    float qr[R], kr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        qr[r] = conv_tap(cs, conv_w, conv_b, float(xr[cq + r]), cq + r, N, K, has_bias, silu);
        kr[r] = conv_tap(cs, conv_w, conv_b, float(xr[ck + r]), ck + r, N, K, has_bias, silu);
    }
    float4 xv = ld4(xr + cv), vv;
#pragma unroll
    for (int j = 0; j < 4; ++j) vv[j] = conv_tap(cs, conv_w, conv_b, xv[j], cv + j, N, K, has_bias, silu);

    // Gates.
    float av = float(a[b * HV + h]) + dt_bias[h];
    float softplus = av > 20.0f ? av : log(1.0f + exp(av));  // torch's softplus threshold
    float decay = exp(-exp(A_log[h]) * softplus);
    float bt = 1.0f / (1.0f + exp(-float(bg[b * HV + h])));

    float4 o = delta_step<R, RS>(s, kr, qr, vv, decay, bt, l2norm, DK);
    if (rs == 0) st4(out + ulong(b * HV + h) * DV + col, o);
#pragma unroll
    for (int r = 0; r < R; ++r) st4(st + r * DV, s[r]);

    // v conv state: this column group is its only reader (all RS lanes read it above, in lockstep).
    simdgroup_barrier(mem_flags::mem_device);
    if (rs == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) conv_shift(cs + (cv + j) * N, T(xv[j]), N);
    }

    // q/k conv state: shifted by the key head's last simdgroup. Every earlier one has consumed its conv inputs
    // (they fed the stores above) before it arrives.
    int arrived = 0;
    if (lane == 0) arrived = atomic_fetch_add_explicit(&arrivals[b * HK + hk], 1, memory_order_relaxed);
    arrived = simd_broadcast(arrived, 0);
    if (arrived == (HV / HK) * blocks - 1) {
        for (int i = lane; i < 2 * DK; i += 32) {
            int c = (i < DK ? 0 : HK * DK - DK) + hk * DK + i;
            conv_shift(cs + c * N, xr[c], N);
        }
        if (lane == 0) atomic_store_explicit(&arrivals[b * HK + hk], 0, memory_order_relaxed);
    }
}

// ------------------------------------------------------------------------------------------ instantiations

#define CONV(T) \
template [[host_name("causal_conv1d_fwd_" #T)]] kernel void causal_conv1d_fwd<T>(device const T*, device const T*, \
    device const T*, device T*, constant int&, constant int&, constant int&, constant int&, constant int&, \
    constant int&, constant int&, constant int&, constant int&, constant int&, constant int&, constant int&, uint3);
#define STEP(T, S) \
template [[host_name("causal_conv1d_step_" #T "_" #S)]] kernel void causal_conv1d_step<T, S>(device const T*, \
    device S*, device const T*, device const T*, device T*, constant int&, constant int&, constant int&, \
    constant int&, constant int&, uint2);
#define GDR(T, S, D, RS) \
template [[host_name("gated_delta_rule_" #T "_" #S "_" #D "_" #RS)]] kernel void gated_delta_rule<T, S, D, RS>( \
    device const T*, device const T*, device const T*, device const float*, device const T*, device S*, device T*, \
    constant int&, constant int&, constant int&, constant int&, constant int&, constant int&, constant int&, \
    constant int&, constant int&, constant int&, constant int&, uint, uint, uint, uint2);
#define GDRC(T, S, D, RS) \
template [[host_name("gated_delta_rule_chunked_" #T "_" #S "_" #D "_" #RS)]] kernel void \
gated_delta_rule_chunked<T, S, D, RS>(device const T*, device const T*, device const T*, device const float*, \
    device const T*, device S*, device T*, constant int&, constant int&, constant int&, constant int&, constant int&, \
    constant int&, constant int&, constant int&, constant int&, constant int&, constant int&, uint, uint, uint, uint2);
#define GDN(T, S, D, RS) \
template [[host_name("gated_delta_net_decode_" #T "_" #S "_" #D "_" #RS)]] kernel void \
gated_delta_net_decode<T, S, D, RS>(device const T*, device T*, device const T*, device const T*, device const T*, \
    device const T*, device const float*, device const float*, device S*, device T*, device atomic_int*, \
    constant int&, constant int&, constant int&, constant int&, constant int&, constant int&, constant int&, \
    constant int&, constant int&, uint, uint, uint, uint2);

CONV(float) CONV(half) CONV(bfloat)
STEP(float, float) STEP(half, half) STEP(half, float) STEP(bfloat, bfloat) STEP(bfloat, float)

// RS (row slices per state column) is chosen by the host per path (see linear_attention.py); the other values
// are kept for `bench.py --sweep`. DK / RS must be a multiple of 4.
#define GDR_TYPES(D, RS) GDR(float, float, D, RS) GDR(half, float, D, RS) GDR(bfloat, float, D, RS) \
    GDR(half, half, D, RS) GDR(bfloat, bfloat, D, RS)
#define GDRC_TYPES(D, RS) GDRC(float, float, D, RS) GDRC(half, float, D, RS) GDRC(bfloat, float, D, RS) \
    GDRC(half, half, D, RS) GDRC(bfloat, bfloat, D, RS)
#define GDN_TYPES(D, RS) GDN(float, float, D, RS) GDN(half, float, D, RS) GDN(bfloat, float, D, RS) \
    GDN(half, half, D, RS) GDN(bfloat, bfloat, D, RS)
GDR_TYPES(64, 16) GDR_TYPES(128, 8) GDR_TYPES(128, 16) GDR_TYPES(128, 32) GDR_TYPES(256, 16)
GDRC_TYPES(64, 16) GDRC_TYPES(128, 8) GDRC_TYPES(128, 16) GDRC_TYPES(128, 32) GDRC_TYPES(256, 32)
GDN_TYPES(64, 8) GDN_TYPES(128, 8) GDN_TYPES(128, 16) GDN_TYPES(128, 32) GDN_TYPES(256, 16)
