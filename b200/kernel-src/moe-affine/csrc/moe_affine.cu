// Batch-small MoE expert matvecs on the MetalExperts affine 4-bit layout (codes LSB-first in uint32 words,
// per-group bf16 scales / biases), for sm_100. One warp per output row; coalesced word loads; x read 16 bytes at a time.
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace {

constexpr int WARPS = 8;

// float(c) of the t-th 4-bit code of w, exactly, without a (quarter-rate) int->float conversion: 2^23 + c - 2^23
__device__ __forceinline__ float code_f(uint32_t w, int t) {
  return __uint_as_float(0x4B000000u | ((w >> (4 * t)) & 0xFu)) - 8388608.f;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// out[o, n] = sum_{j<J} rw(r) * dot(x[r / x_div], dequant(w[ids[r], n])),  r = o * J + j,  rw = weights ? weights[r] : 1
// A warp owns ROWS consecutive output rows; its lanes stride over the flattened (route, 32-code chunk) space so every
// route's loads are in flight together. One uint4 = 4 words = 32 codes, inside one group (G >= 32).
template <int G, int ROWS>
__global__ void __launch_bounds__(WARPS * 32) qmv4_kernel(
    const __nv_bfloat16* __restrict__ x, const uint4* __restrict__ w, const __nv_bfloat16* __restrict__ s,
    const __nv_bfloat16* __restrict__ b, const int* __restrict__ ids, const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ out, int N, int K, int J, int x_div, long w_se, long w_sn, long s_se, long s_sn,
    long b_se, long b_sn) {
  const int lane = threadIdx.x & 31, o = blockIdx.y;
  const int n0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * ROWS;
  if (n0 >= N) return;
  const int CH = K / 32;  // uint4 chunks per row
  float acc[ROWS];
#pragma unroll
  for (int k = 0; k < ROWS; ++k) acc[k] = 0.f;
#pragma unroll 2
  for (int i = lane; i < J * CH; i += 32) {
    const int j = i / CH, c = i - j * CH, r = o * J + j;
    const float rw = weights ? __ldg(weights + r) : 1.f;
    if (rw == 0.f) continue;
    const long e = __ldg(ids + r);
    const uint4* xp = reinterpret_cast<const uint4*>(x + (long)(r / x_div) * K) + c * 4;
    float xf[32];
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      uint4 xv = __ldg(xp + v);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
      for (int p = 0; p < 4; ++p) { float2 f = __bfloat1622float2(h[p]); xf[v * 8 + 2 * p] = f.x; xf[v * 8 + 2 * p + 1] = f.y; }
    }
    float sx = 0.f;
#pragma unroll
    for (int q = 0; q < 32; ++q) sx += xf[q];
    const int g = (c * 32) / G;
    uint4 qv[ROWS];
#pragma unroll
    for (int k = 0; k < ROWS; ++k) qv[k] = __ldg(w + (e * w_se + (long)(n0 + k) * w_sn) / 4 + c);
#pragma unroll
    for (int k = 0; k < ROWS; ++k) {
      const uint32_t words[4] = {qv[k].x, qv[k].y, qv[k].z, qv[k].w};
      float cx = 0.f;
#pragma unroll
      for (int u = 0; u < 4; ++u)
#pragma unroll
        for (int t = 0; t < 8; ++t) cx = fmaf(code_f(words[u], t), xf[u * 8 + t], cx);
      const long row = n0 + k;
      const float sc = __bfloat162float(__ldg(s + e * s_se + row * s_sn + g));
      const float bi = __bfloat162float(__ldg(b + e * b_se + row * b_sn + g));
      acc[k] = fmaf(rw, fmaf(sc, cx, bi * sx), acc[k]);
    }
  }
#pragma unroll
  for (int k = 0; k < ROWS; ++k) {
    float v = warp_sum(acc[k]);
    if (lane == 0 && n0 + k < N) out[(long)o * N + n0 + k] = __float2bfloat16(v);
  }
}

template <int ROWS>
void launch(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s, const torch::Tensor& b,
            const torch::Tensor& ids, const float* weights, torch::Tensor& out, int64_t O, int64_t J, int64_t x_div,
            int64_t G) {
  const int64_t K = x.size(-1), N = w.size(1);
  TORCH_CHECK(N % ROWS == 0 && w.stride(1) % 4 == 0 && w.stride(0) % 4 == 0, "rows / 16-byte alignment");
  dim3 grid((N / ROWS + WARPS - 1) / WARPS, O);
  auto stream = at::cuda::getCurrentCUDAStream();
#define L(GS) qmv4_kernel<GS, ROWS><<<grid, WARPS * 32, 0, stream>>>( \
      (const __nv_bfloat16*)x.data_ptr(), (const uint4*)w.data_ptr(), (const __nv_bfloat16*)s.data_ptr(), \
      (const __nv_bfloat16*)b.data_ptr(), ids.data_ptr<int>(), weights, (__nv_bfloat16*)out.data_ptr(), N, K, J, x_div, \
      w.stride(0), w.stride(1), s.stride(0), s.stride(1), b.stride(0), b.stride(1))
  if (G == 32) L(32); else if (G == 64) L(64); else L(128);
#undef L
}

void check(const torch::Tensor& w, const torch::Tensor& s, const torch::Tensor& b, int64_t K, int64_t G) {
  TORCH_CHECK(w.stride(-1) == 1 && s.stride(-1) == 1 && b.stride(-1) == 1, "banks: last dim must be contiguous");
  TORCH_CHECK(s.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16, "scales/biases must be bf16");
  TORCH_CHECK(K % 64 == 0 && w.size(-1) * 8 == K, "K must be a multiple of 64 and match the packed width");
  TORCH_CHECK(G == 32 || G == 64 || G == 128, "group size 32/64/128");
}

#define UNUSED_DISPATCH_G(G, ...)                         \
  switch (G) {                                     \
    case 32: { constexpr int GS = 32; __VA_ARGS__; break; } \
    case 64: { constexpr int GS = 64; __VA_ARGS__; break; } \
    default: { constexpr int GS = 128; __VA_ARGS__; break; } \
  }

}  // namespace

torch::Tensor gather_qmv4(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor b, torch::Tensor ids,
                          int64_t x_div, int64_t G) {
  const int64_t K = x.size(-1), N = w.size(1), R = ids.numel();
  check(w, s, b, K, G);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && ids.scalar_type() == at::kInt && ids.is_contiguous());
  auto out = torch::empty({R, N}, x.options());
  if (R) launch<4>(x, w, s, b, ids, nullptr, out, R, 1, x_div, G);
  return out;
}

torch::Tensor down_route4(torch::Tensor act, torch::Tensor w, torch::Tensor s, torch::Tensor b, torch::Tensor ids,
                          torch::Tensor weights, int64_t topk, int64_t G) {
  const int64_t K = act.size(-1), N = w.size(1), R = ids.numel(), T = R / topk;
  check(w, s, b, K, G);
  TORCH_CHECK(act.scalar_type() == at::kBFloat16 && act.is_contiguous() && weights.scalar_type() == at::kFloat && weights.is_contiguous());
  auto out = torch::empty({T, N}, act.options());
  if (T) launch<1>(act, w, s, b, ids, weights.data_ptr<float>(), out, T, topk, 1, G);
  return out;
}


// ---- bf16 small-M GEMV: out[m, n] = x[m] . W[n] (+ bias[n]), M <= 8 --------------------------------------------
template <int M>
__device__ __forceinline__ void dot8(const uint4 wv, const __nv_bfloat16* __restrict__ x, long ldx, int c, float* acc) {
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
  float wf[8];
#pragma unroll
  for (int p = 0; p < 4; ++p) { float2 f = __bfloat1622float2(w2[p]); wf[2 * p] = f.x; wf[2 * p + 1] = f.y; }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + m * ldx) + c);
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      float2 f = __bfloat1622float2(x2[p]);
      acc[m] = fmaf(wf[2 * p], f.x, fmaf(wf[2 * p + 1], f.y, acc[m]));
    }
  }
}

// One warp per ROWS rows, lanes over K in 8-element chunks
template <int M, int ROWS>
__global__ void __launch_bounds__(256) gemv_warp_kernel(const __nv_bfloat16* __restrict__ x, const uint4* __restrict__ W,
                                                        const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
                                                        int N, int K, long ldw) {
  const int lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * ROWS;
  if (n0 >= N) return;
  const int CH = K / 8;
  float acc[ROWS][M];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
#pragma unroll 2
  for (int c = lane; c < CH; c += 32) {
    uint4 wv[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) wv[r] = (n0 + r < N) ? __ldg(W + ((long)(n0 + r) * ldw) / 8 + c) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) dot8<M>(wv[r], x, K, c, acc[r]);
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = warp_sum(acc[r][m]);
      if (lane == 0 && n0 + r < N) {
        if (bias) v += __bfloat162float(bias[n0 + r]);
        out[(long)m * N + n0 + r] = __float2bfloat16(v);
      }
    }
}

// One block per row: 256 threads over K (short N, long K)
template <int M>
__global__ void __launch_bounds__(256) gemv_block_kernel(const __nv_bfloat16* __restrict__ x, const uint4* __restrict__ W,
                                                         const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
                                                         int N, int K, long ldw) {
  __shared__ float part[8][M];
  const int n = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;
  for (int c = threadIdx.x; c < K / 8; c += 256) dot8<M>(__ldg(W + ((long)n * ldw) / 8 + c), x, K, c, acc);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float v = warp_sum(acc[m]);
    if (lane == 0) part[warp][m] = v;
  }
  __syncthreads();
  if (threadIdx.x < M) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) v += part[w][threadIdx.x];
    if (bias) v += __bfloat162float(bias[n]);
    out[(long)threadIdx.x * N + n] = __float2bfloat16(v);
  }
}

template <int M>
void gemv_launch(const torch::Tensor& x, const torch::Tensor& W, const __nv_bfloat16* bias, torch::Tensor& out) {
  const int64_t N = W.size(0), K = W.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = (const __nv_bfloat16*)x.data_ptr();
  const auto* wp = (const uint4*)W.data_ptr();
  auto* op = (__nv_bfloat16*)out.data_ptr();
  if (N < 512) {
    gemv_block_kernel<M><<<N, 256, 0, stream>>>(xp, wp, bias, op, N, K, W.stride(0));
  } else if (K <= 1024) {
    gemv_warp_kernel<M, 4><<<(N / 4 + 7) / 8 + 1, 256, 0, stream>>>(xp, wp, bias, op, N, K, W.stride(0));
  } else {
    gemv_warp_kernel<M, 2><<<(N / 2 + 7) / 8 + 1, 256, 0, stream>>>(xp, wp, bias, op, N, K, W.stride(0));
  }
}

torch::Tensor gemv_bf16(torch::Tensor x, torch::Tensor W, std::optional<torch::Tensor> bias) {
  const int64_t K = x.size(-1), N = W.size(0), M = x.numel() / K;
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && W.scalar_type() == at::kBFloat16 && x.is_contiguous(), "bf16, contiguous x");
  TORCH_CHECK(W.size(1) == K && W.stride(1) == 1 && W.stride(0) % 8 == 0 && K % 8 == 0, "W [N, K] row-major, K % 8 == 0");
  TORCH_CHECK(M == 1 || M == 2 || M == 3 || M == 4 || M == 8, "M in {1, 2, 3, 4, 8}");
  auto sizes = x.sizes().vec(); sizes.back() = N;
  auto out = torch::empty(sizes, x.options());
  const __nv_bfloat16* bp = bias ? (const __nv_bfloat16*)bias->data_ptr() : nullptr;
  switch (M) {
    case 1: gemv_launch<1>(x, W, bp, out); break;
    case 2: gemv_launch<2>(x, W, bp, out); break;
    case 3: gemv_launch<3>(x, W, bp, out); break;
    case 4: gemv_launch<4>(x, W, bp, out); break;
    default: gemv_launch<8>(x, W, bp, out); break;
  }
  return out;
}

// ---- GPU-driven expert offload ---------------------------------------------------------------------------------
// Experts live in pinned host memory (UVA: device-addressable) and a device pool of S slots caches them. `table[l*E+e]`
// is the slot of expert e of layer l (-1: not resident), `owner[s]` the l*E+e a slot holds (-1: free), `age[s]` its
// last use. One token per call (decode): its routes are distinct experts.
__global__ void offload_plan_kernel(const int* __restrict__ ids, int R, int layer, int E, int* __restrict__ table,
                                    int* __restrict__ owner, long* __restrict__ age, long* __restrict__ clock, int S,
                                    int* __restrict__ slots, int* __restrict__ miss) {
  __shared__ long best_age[32];
  __shared__ int best_slot[32];
  __shared__ int route_slot[64];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarps = blockDim.x >> 5;
  const long now = *clock + 1;
  if (tid < R) {
    const int key = layer * E + ids[tid];
    const int s = table[key];
    route_slot[tid] = s;
    if (s >= 0) age[s] = now;  // hits are newest: never victims for this call's misses
  }
  __syncthreads();
  for (int r = 0; r < R; ++r) {
    if (route_slot[r] >= 0) continue;  // uniform across the block (shared)
    // oldest slot (free slots have age -1)
    long a = 0x7fffffffffffffffL; int sl = -1;
    for (int i = tid; i < S; i += blockDim.x) { long v = age[i]; if (v < a) { a = v; sl = i; } }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      long oa = __shfl_xor_sync(0xffffffffu, a, o); int os = __shfl_xor_sync(0xffffffffu, sl, o);
      if (oa < a || (oa == a && os < sl)) { a = oa; sl = os; }
    }
    if (lane == 0) { best_age[warp] = a; best_slot[warp] = sl; }
    __syncthreads();
    if (tid == 0) {
      const int key0 = layer * E + ids[r];
      const int existing = table[key0];  // several tokens per call: an earlier route may have installed this expert
      if (existing >= 0) {
        route_slot[r] = existing | 0x40000000;  // read from the host again (same bytes into the same slot)
        age[existing] = now;
        goto next_route;
      }
      {
      long ba = best_age[0]; int bs = best_slot[0];
      for (int w = 1; w < nwarps; ++w) if (best_age[w] < ba || (best_age[w] == ba && best_slot[w] < bs)) { ba = best_age[w]; bs = best_slot[w]; }
      const int old = owner[bs];
      if (old >= 0) table[old] = -1;
      const int key = layer * E + ids[r];
      owner[bs] = key; table[key] = bs; age[bs] = now;
      route_slot[r] = bs | 0x40000000;  // tagged: a miss installed into this slot
      }
    next_route:;
    }
    __syncthreads();
  }
  if (tid < R) {
    const int v = route_slot[tid];
    slots[tid] = v & 0x3fffffff;
    miss[tid] = (v >> 30) & 1;
  }
  if (tid == 0) *clock = now;
}

// The expert matvec over pool / host sources: route r reads slot `slots[r]` of the pool, or (miss) expert `ids[r]` of
// the host banks and writes what it reads into that slot.
template <int G, int ROWS>
__global__ void __launch_bounds__(WARPS * 32) qmv4_offload_kernel(
    const __nv_bfloat16* __restrict__ x, uint4* pw, __nv_bfloat16* ps, __nv_bfloat16* pb,
    const uint4* __restrict__ hw, const __nv_bfloat16* __restrict__ hs, const __nv_bfloat16* __restrict__ hb,
    const int* __restrict__ ids, const int* __restrict__ slots, const int* __restrict__ miss,
    const float* __restrict__ weights, __nv_bfloat16* __restrict__ out, int N, int K, int J, int x_div,
    long w_se, long w_sn, long s_se, long s_sn) {
  const int lane = threadIdx.x & 31, o = blockIdx.y;
  const int n0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * ROWS;
  if (n0 >= N) return;
  const int CH = K / 32;
  float acc[ROWS];
#pragma unroll
  for (int k = 0; k < ROWS; ++k) acc[k] = 0.f;
#pragma unroll 2
  for (int i = lane; i < J * CH; i += 32) {
    const int j = i / CH, c = i - j * CH, r = o * J + j;
    const float rw = weights ? __ldg(weights + r) : 1.f;
    if (rw == 0.f) continue;
    const bool m = miss[r];
    const long src_e = m ? (long)ids[r] : (long)slots[r], dst_e = slots[r];
    const uint4* wsrc = m ? hw : pw;
    const __nv_bfloat16* ssrc = m ? hs : ps;
    const __nv_bfloat16* bsrc = m ? hb : pb;
    const uint4* xp = reinterpret_cast<const uint4*>(x + (long)(r / x_div) * K) + c * 4;
    float xf[32];
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      uint4 xv = __ldg(xp + v);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
      for (int p = 0; p < 4; ++p) { float2 f = __bfloat1622float2(h[p]); xf[v * 8 + 2 * p] = f.x; xf[v * 8 + 2 * p + 1] = f.y; }
    }
    float sx = 0.f;
#pragma unroll
    for (int q = 0; q < 32; ++q) sx += xf[q];
    const int g = (c * 32) / G;
    const bool owns_group = ((c * 32) % G) == 0;  // one chunk per group copies the scale / bias
    uint4 qv[ROWS];
#pragma unroll
    for (int k = 0; k < ROWS; ++k) qv[k] = wsrc[(src_e * w_se + (long)(n0 + k) * w_sn) / 4 + c];
#pragma unroll
    for (int k = 0; k < ROWS; ++k) {
      const uint32_t words[4] = {qv[k].x, qv[k].y, qv[k].z, qv[k].w};
      float cx = 0.f;
#pragma unroll
      for (int u = 0; u < 4; ++u)
#pragma unroll
        for (int t = 0; t < 8; ++t) cx = fmaf(code_f(words[u], t), xf[u * 8 + t], cx);
      const long row = n0 + k;
      const __nv_bfloat16 sv = ssrc[src_e * s_se + row * s_sn + g], bv = bsrc[src_e * s_se + row * s_sn + g];
      if (m) {
        pw[(dst_e * w_se + row * w_sn) / 4 + c] = qv[k];
        if (owns_group) { ps[dst_e * s_se + row * s_sn + g] = sv; pb[dst_e * s_se + row * s_sn + g] = bv; }
      }
      acc[k] = fmaf(rw, fmaf(__bfloat162float(sv), cx, __bfloat162float(bv) * sx), acc[k]);
    }
  }
#pragma unroll
  for (int k = 0; k < ROWS; ++k) {
    float v = warp_sum(acc[k]);
    if (lane == 0 && n0 + k < N) out[(long)o * N + n0 + k] = __float2bfloat16(v);
  }
}

std::tuple<torch::Tensor, torch::Tensor> offload_plan(torch::Tensor ids, int64_t layer, int64_t E, torch::Tensor table,
                                        torch::Tensor owner, torch::Tensor age, torch::Tensor clock) {
  const int R = ids.numel(), S = owner.numel();
  TORCH_CHECK(R <= 64 && ids.scalar_type() == at::kInt && table.scalar_type() == at::kInt && age.scalar_type() == at::kLong);
  auto slots = torch::empty({R}, ids.options()), miss = torch::empty({R}, ids.options());
  offload_plan_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
      ids.data_ptr<int>(), R, layer, E, table.data_ptr<int>(), owner.data_ptr<int>(), age.data_ptr<int64_t>(),
      clock.data_ptr<int64_t>(), S, slots.data_ptr<int>(), miss.data_ptr<int>());
  return std::make_tuple(slots, miss);
}

// pool banks are slot-indexed [S, N, ...], host banks expert-indexed [E, N, ...], same row layout
torch::Tensor qmv4_offload(torch::Tensor x, torch::Tensor pw, torch::Tensor ps, torch::Tensor pb, int64_t hw,
                           int64_t hs, int64_t hb, torch::Tensor ids, torch::Tensor slots, torch::Tensor miss,
                           std::optional<torch::Tensor> weights, int64_t J, int64_t x_div, int64_t G) {
  const int64_t K = x.size(-1), N = pw.size(1), R = ids.numel(), O = R / J;
  TORCH_CHECK(pb.stride(0) == ps.stride(0) && pb.stride(1) == ps.stride(1), "scales / biases layouts differ");
  auto out = torch::empty({O, N}, x.options());
  const float* wp = weights ? weights->data_ptr<float>() : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream();
#define LO(GS, RW) qmv4_offload_kernel<GS, RW><<<dim3((N / RW + WARPS - 1) / WARPS, O), WARPS * 32, 0, stream>>>( \
      (const __nv_bfloat16*)x.data_ptr(), (uint4*)pw.data_ptr(), (__nv_bfloat16*)ps.data_ptr(), (__nv_bfloat16*)pb.data_ptr(), \
      (const uint4*)hw, (const __nv_bfloat16*)hs, (const __nv_bfloat16*)hb, \
      ids.data_ptr<int>(), slots.data_ptr<int>(), miss.data_ptr<int>(), wp, (__nv_bfloat16*)out.data_ptr(), N, K, J, x_div, \
      pw.stride(0), pw.stride(1), ps.stride(0), ps.stride(1))
  if (J == 1) { TORCH_CHECK(N % 4 == 0); if (G == 64) LO(64, 4); else if (G == 32) LO(32, 4); else LO(128, 4); }
  else { if (G == 64) LO(64, 1); else if (G == 32) LO(32, 1); else LO(128, 1); }
#undef LO
  return out;
}

// Copy the missed experts (miss[r] != 0) of up to 6 banks from host (UVA) into their pool slots. Grid: (chunks, R).
struct Banks6 { const uint4* host[6]; uint4* pool[6]; long expert_vecs[6]; };  // per-expert size in uint4
__global__ void __launch_bounds__(256) offload_install_kernel(const int* __restrict__ ids, const int* __restrict__ slots,
                                                              const int* __restrict__ miss, Banks6 banks, int nbanks) {
  const int r = blockIdx.y;
  if (!miss[r]) return;
  const long e = ids[r], s = slots[r];
  for (int b = 0; b < nbanks; ++b) {
    const long n = banks.expert_vecs[b];
    const uint4* src = banks.host[b] + e * n;
    uint4* dst = banks.pool[b] + s * n;
    for (long i = blockIdx.x * 256L + threadIdx.x; i < n; i += 256L * gridDim.x) dst[i] = src[i];
  }
}

void offload_install(torch::Tensor ids, torch::Tensor slots, torch::Tensor miss, std::vector<int64_t> host,
                     std::vector<torch::Tensor> pool, int64_t blocks) {
  TORCH_CHECK(host.size() == pool.size() && pool.size() <= 6);
  Banks6 b{};
  for (size_t i = 0; i < pool.size(); ++i) {
    TORCH_CHECK(pool[i].is_contiguous() && (pool[i][0].numel() * pool[i].element_size()) % 16 == 0);
    b.host[i] = reinterpret_cast<const uint4*>(host[i]);
    b.pool[i] = reinterpret_cast<uint4*>(pool[i].data_ptr());
    b.expert_vecs[i] = pool[i][0].numel() * pool[i].element_size() / 16;
  }
  const int R = ids.numel();
  if (R == 0) return;
  offload_install_kernel<<<dim3(blocks, R), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      ids.data_ptr<int>(), slots.data_ptr<int>(), miss.data_ptr<int>(), b, (int)pool.size());
}

// ---- Router: logits = x @ W.T (bf16-rounded like F.linear), softmax (fp32), top-k, optional renormalization --------
// One token. Warps compute rows across the grid; the last block to finish does softmax + top-k in one warp.

template <int MAXE>
__global__ void __launch_bounds__(256) router_topk_kernel(const __nv_bfloat16* __restrict__ x, const uint4* __restrict__ W,
                                                          float* __restrict__ scratch, __nv_bfloat16* __restrict__ logits_out,
                                                          __nv_bfloat16* __restrict__ wout, int64_t* __restrict__ iout,
                                                          int E, int K, int k, bool norm,
                                                          unsigned int* __restrict__ counter) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 8 + warp, tk = blockIdx.y;
  x += (long)tk * K; scratch += (long)tk * E; logits_out += (long)tk * E; wout += (long)tk * k; iout += (long)tk * k;
  counter += tk;
  if (row < E) {
    float acc[1] = {0.f};
    for (int c = lane; c < K / 8; c += 32) dot8<1>(__ldg(W + (long)row * (K / 8) + c), x, K, c, acc);
    float v = warp_sum(acc[0]);
    if (lane == 0) {
      __nv_bfloat16 b = __float2bfloat16(v);
      logits_out[row] = b;
      scratch[row] = __bfloat162float(b);
    }
  }
  __shared__ bool last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(counter, 1) == gridDim.x - 1;
  __syncthreads();
  if (!last || warp) return;
  __threadfence();
  constexpr int PER = MAXE / 32;
  float vals[PER];
  float mx = -INFINITY;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int e = lane + 32 * i;
    vals[i] = e < E ? __ldcg(scratch + e) : -INFINITY;
    mx = fmaxf(mx, vals[i]);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  float sum = 0.f;
#pragma unroll
  for (int i = 0; i < PER; ++i) { vals[i] = (lane + 32 * i) < E ? __expf(vals[i] - mx) : 0.f; sum += vals[i]; }
  sum = warp_sum(sum);
  float picked[16]; int pidx[16]; float tot = 0.f;
  for (int j = 0; j < k; ++j) {
    float bv = -1.f; int bi = 0x7fffffff;
#pragma unroll
    for (int i = 0; i < PER; ++i) { const int e = lane + 32 * i; if (vals[i] > bv || (vals[i] == bv && e < bi)) { bv = vals[i]; bi = e; } }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffffu, bv, o); int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((bi & 31) == lane) vals[bi >> 5] = -2.f;  // remove the winner
    picked[j] = bv / sum; pidx[j] = bi; tot += picked[j];
  }
  if (lane == 0) {
    for (int j = 0; j < k; ++j) {
      wout[j] = __float2bfloat16(norm ? picked[j] / tot : picked[j]);
      iout[j] = pidx[j];
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> router_topk(torch::Tensor x, torch::Tensor W, int64_t k, bool norm) {
  const int64_t E = W.size(0), K = W.size(1), T = x.numel() / K;
  TORCH_CHECK(x.numel() == T * K && T >= 1 && T <= 8 && x.scalar_type() == at::kBFloat16 && W.scalar_type() == at::kBFloat16 && W.is_contiguous() && x.is_contiguous());
  TORCH_CHECK(E <= 1024 && k <= 16 && K % 8 == 0, "E <= 1024, k <= 16");
  auto opts = x.options();
  auto scratch = torch::empty({T, E}, opts.dtype(at::kFloat));
  // per-call, per-token "last block" counters (a global one breaks when two calls overlap, e.g. on two streams)
  auto counter = torch::zeros({T}, opts.dtype(at::kInt));
  auto logits = torch::empty({T, E}, opts);
  auto w = torch::empty({T, k}, opts);
  auto idx = torch::empty({T, k}, opts.dtype(at::kLong));
  const dim3 blocks((E + 7) / 8, T);
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = (const __nv_bfloat16*)x.data_ptr();
  const auto* wp = (const uint4*)W.data_ptr();
  if (E <= 512)
    router_topk_kernel<512><<<blocks, 256, 0, stream>>>(xp, wp, scratch.data_ptr<float>(), (__nv_bfloat16*)logits.data_ptr(), (__nv_bfloat16*)w.data_ptr(), idx.data_ptr<int64_t>(), E, K, k, norm, (unsigned int*)counter.data_ptr<int>());
  else
    router_topk_kernel<1024><<<blocks, 256, 0, stream>>>(xp, wp, scratch.data_ptr<float>(), (__nv_bfloat16*)logits.data_ptr(), (__nv_bfloat16*)w.data_ptr(), idx.data_ptr<int64_t>(), E, K, k, norm, (unsigned int*)counter.data_ptr<int>());
  return std::make_tuple(logits, w, idx);
}

// ---- v2: route metadata preloaded per warp (shuffled), every weight / scale load of a lane issued before the math ---
// Route r = o * J + j reads expert `src` of `wsrc` (host if missed, else pool slot), installs into pool slot `dst` if
// missed. Plain (resident) calls: miss == nullptr, slots == nullptr, the pool is the bank and `ids` index it.
template <int G, int ROWS, int MAXIT>
__global__ void __launch_bounds__(WARPS * 32) qmv4_v2_kernel(
    const __nv_bfloat16* __restrict__ x, uint4* pw, __nv_bfloat16* ps, __nv_bfloat16* pb,
    const uint4* __restrict__ hw, const __nv_bfloat16* __restrict__ hs, const __nv_bfloat16* __restrict__ hb,
    const int* __restrict__ ids, const int* __restrict__ slots, const int* __restrict__ miss,
    const float* __restrict__ weights, __nv_bfloat16* __restrict__ out, int N, int K, int J, int x_div,
    long w_se, long w_sn, long s_se, long s_sn) {
  const int lane = threadIdx.x & 31, o = blockIdx.y;
  const int n0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * ROWS;
  if (n0 >= N) return;
  const int CH = K / 32;
  // metadata of route j held by lane j
  int m_src = 0, m_dst = 0, m_miss = 0; float m_rw = 0.f;
  if (lane < J) {
    const int r = o * J + lane;
    m_miss = miss ? miss[r] : 0;
    m_dst = slots ? slots[r] : ids[r];
    m_src = m_miss ? ids[r] : m_dst;
    m_rw = weights ? weights[r] : 1.f;
  }
  uint4 qv[MAXIT][ROWS];
  __nv_bfloat16 sv[MAXIT][ROWS], bv[MAXIT][ROWS];
  int jj[MAXIT];
#pragma unroll
  for (int it = 0; it < MAXIT; ++it) {
    const int i = lane + 32 * it;
    const int j = i / CH, c = i - j * CH;
    const int src = __shfl_sync(0xffffffffu, m_src, j & 31), mi = __shfl_sync(0xffffffffu, m_miss, j & 31);
    const float rw = __shfl_sync(0xffffffffu, m_rw, j & 31);
    jj[it] = (i < J * CH && rw != 0.f) ? j : -1;
    if (jj[it] >= 0) {
      const uint4* wsrc = mi ? hw : pw;
      const __nv_bfloat16* ssrc = mi ? hs : ps;
      const __nv_bfloat16* bsrc = mi ? hb : pb;
      const int g = (c * 32) / G;
#pragma unroll
      for (int k = 0; k < ROWS; ++k) {
        const long row = n0 + k;
        qv[it][k] = wsrc[((long)src * w_se + row * w_sn) / 4 + c];
        sv[it][k] = ssrc[(long)src * s_se + row * s_sn + g];
        bv[it][k] = bsrc[(long)src * s_se + row * s_sn + g];
      }
    }
  }
  float acc[ROWS];
#pragma unroll
  for (int k = 0; k < ROWS; ++k) acc[k] = 0.f;
#pragma unroll
  for (int it = 0; it < MAXIT; ++it) {
    const int i = lane + 32 * it;
    const int j = jj[it];
    const int jsafe = j < 0 ? 0 : j;
    const int dst = __shfl_sync(0xffffffffu, m_dst, jsafe & 31), mi = __shfl_sync(0xffffffffu, m_miss, jsafe & 31);
    const float rw = __shfl_sync(0xffffffffu, m_rw, jsafe & 31);
    if (j < 0) continue;
    const int c = i - j * CH, r = o * J + j;
    const uint4* xp = reinterpret_cast<const uint4*>(x + (long)(r / x_div) * K) + c * 4;
    float xf[32];
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      uint4 xv = __ldg(xp + v);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
      for (int p = 0; p < 4; ++p) { float2 f = __bfloat1622float2(h[p]); xf[v * 8 + 2 * p] = f.x; xf[v * 8 + 2 * p + 1] = f.y; }
    }
    float sx = 0.f;
#pragma unroll
    for (int q = 0; q < 32; ++q) sx += xf[q];
    const int g = (c * 32) / G;
    const bool owns_group = ((c * 32) % G) == 0;
#pragma unroll
    for (int k = 0; k < ROWS; ++k) {
      const uint32_t words[4] = {qv[it][k].x, qv[it][k].y, qv[it][k].z, qv[it][k].w};
      float cx = 0.f;
#pragma unroll
      for (int u = 0; u < 4; ++u)
#pragma unroll
        for (int t = 0; t < 8; ++t) cx = fmaf(code_f(words[u], t), xf[u * 8 + t], cx);
      if (mi) {
        const long row = n0 + k;
        pw[((long)dst * w_se + row * w_sn) / 4 + c] = qv[it][k];
        if (owns_group) { ps[(long)dst * s_se + row * s_sn + g] = sv[it][k]; pb[(long)dst * s_se + row * s_sn + g] = bv[it][k]; }
      }
      acc[k] = fmaf(rw, fmaf(__bfloat162float(sv[it][k]), cx, __bfloat162float(bv[it][k]) * sx), acc[k]);
    }
  }
#pragma unroll
  for (int k = 0; k < ROWS; ++k) {
    float v = warp_sum(acc[k]);
    if (lane == 0 && n0 + k < N) out[(long)o * N + n0 + k] = __float2bfloat16(v);
  }
}

template <int ROWS, int MAXIT>
void v2_launch(const torch::Tensor& x, const torch::Tensor& pw, const torch::Tensor& ps, const torch::Tensor& pb,
               const void* hw, const void* hs, const void* hb, const torch::Tensor& ids, const int* slots, const int* miss,
               const float* wts, torch::Tensor& out, int64_t O, int64_t J, int64_t x_div, int64_t G) {
  const int64_t K = x.size(-1), N = pw.size(1);
  TORCH_CHECK(N % ROWS == 0 && J * (K / 32) <= 32 * MAXIT, "v2 launch: shape exceeds MAXIT");
  dim3 grid((N / ROWS + WARPS - 1) / WARPS, O);
  auto stream = at::cuda::getCurrentCUDAStream();
#define LV(GS) qmv4_v2_kernel<GS, ROWS, MAXIT><<<grid, WARPS * 32, 0, stream>>>( \
      (const __nv_bfloat16*)x.data_ptr(), (uint4*)pw.data_ptr(), (__nv_bfloat16*)ps.data_ptr(), (__nv_bfloat16*)pb.data_ptr(), \
      (const uint4*)hw, (const __nv_bfloat16*)hs, (const __nv_bfloat16*)hb, ids.data_ptr<int>(), slots, miss, wts, \
      (__nv_bfloat16*)out.data_ptr(), N, K, J, x_div, pw.stride(0), pw.stride(1), ps.stride(0), ps.stride(1))
  if (G == 64) LV(64); else if (G == 32) LV(32); else LV(128);
#undef LV
}

// dispatch on the flattened per-row work (J * K / 32 chunks): gate/up (J = 1, K = 2560: 80) / down (J = top_k, K = 640)
void v2_dispatch(const torch::Tensor& x, const torch::Tensor& pw, const torch::Tensor& ps, const torch::Tensor& pb,
                 const void* hw, const void* hs, const void* hb, const torch::Tensor& ids, const int* slots, const int* miss,
                 const float* wts, torch::Tensor& out, int64_t O, int64_t J, int64_t x_div, int64_t G, int64_t rows) {
  const int64_t work = J * (x.size(-1) / 32);
  if (rows == 4 && work <= 96) v2_launch<4, 3>(x, pw, ps, pb, hw, hs, hb, ids, slots, miss, wts, out, O, J, x_div, G);
  else if (rows == 2 && work <= 96) v2_launch<2, 3>(x, pw, ps, pb, hw, hs, hb, ids, slots, miss, wts, out, O, J, x_div, G);
  else if (rows == 1 && work <= 96) v2_launch<1, 3>(x, pw, ps, pb, hw, hs, hb, ids, slots, miss, wts, out, O, J, x_div, G);
  else if (rows == 2 && work <= 224) v2_launch<2, 7>(x, pw, ps, pb, hw, hs, hb, ids, slots, miss, wts, out, O, J, x_div, G);
  else if (work <= 224) v2_launch<1, 7>(x, pw, ps, pb, hw, hs, hb, ids, slots, miss, wts, out, O, J, x_div, G);
  else if (work <= 512) v2_launch<1, 16>(x, pw, ps, pb, hw, hs, hb, ids, slots, miss, wts, out, O, J, x_div, G);
  else TORCH_CHECK(false, "v2: per-row work too large");
}

torch::Tensor qmv4_offload_v2(torch::Tensor x, torch::Tensor pw, torch::Tensor ps, torch::Tensor pb, int64_t hw, int64_t hs,
                              int64_t hb, torch::Tensor ids, std::optional<torch::Tensor> slots, std::optional<torch::Tensor> miss,
                              std::optional<torch::Tensor> weights, int64_t J, int64_t x_div, int64_t G, int64_t rows) {
  const int64_t N = pw.size(1), R = ids.numel(), O = R / J;
  TORCH_CHECK(J <= 32 && x.is_contiguous() && ids.scalar_type() == at::kInt);
  auto out = torch::empty({O, N}, x.options());
  if (O == 0) return out;
  v2_dispatch(x, pw, ps, pb, (const void*)hw, (const void*)hs, (const void*)hb, ids, slots ? slots->data_ptr<int>() : nullptr,
              miss ? miss->data_ptr<int>() : nullptr, weights ? weights->data_ptr<float>() : nullptr, out, O, J, x_div, G, rows);
  return out;
}

// ---- host compute tier: GPU side ---------------------------------------------------------------------------------
// Lookup only (the pool is not refilled): slot of each route's expert, or -1; hits refresh their age.
__global__ void offload_lookup_kernel(const int* __restrict__ ids, int R, int layer, int E, const int* __restrict__ table,
                                      long* __restrict__ age, const long* __restrict__ clock, int* __restrict__ slots,
                                      float* __restrict__ gpu_w, const float* __restrict__ w) {
  const int r = threadIdx.x;
  if (r >= R) return;
  const int s = table[layer * E + ids[r]];
  slots[r] = s >= 0 ? s : 0;
  gpu_w[r] = s >= 0 ? w[r] : 0.f;
  if (s >= 0) age[s] = *clock;
}

struct HostMailbox { volatile unsigned long long seq; int layer, n, ntok; int ids[64]; float w[64]; int tok[64]; unsigned short x[4 * 8192]; };
struct HostResult { volatile unsigned long long done; float out[4 * 8192]; };

// Post the missed routes (not resident) + the activation to the host tier; state[0]: sequence counter, state[1]: the
// sequence this call waits for (0: nothing posted)
__global__ void host_post_kernel(const int* __restrict__ ids, const int* __restrict__ slots_raw, const float* __restrict__ w,
                                 int R, int layer, int E, const int* __restrict__ table, const __nv_bfloat16* __restrict__ x,
                                 int H, int ntok, HostMailbox* mb, unsigned long long* state) {
  __shared__ int n;
  const int topk = R / ntok;
  if (threadIdx.x == 0) {
    n = 0;
    for (int r = 0; r < R; ++r)
      if (table[layer * E + ids[r]] < 0) { mb->ids[n] = ids[r]; mb->w[n] = w[r]; mb->tok[n] = r / topk; ++n; }
  }
  __syncthreads();
  if (n == 0) { if (threadIdx.x == 0) state[1] = 0; return; }
  const unsigned short* xs = reinterpret_cast<const unsigned short*>(x);
  for (int i = threadIdx.x; i < ntok * H; i += blockDim.x) mb->x[i] = xs[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    mb->layer = layer; mb->n = n; mb->ntok = ntok;
    __threadfence_system();
    const unsigned long long s = state[0] + 1;
    state[0] = s; state[1] = s;
    mb->seq = s;
    __threadfence_system();
  }
}

// out += host result (when a call was posted): one block spins on the done flag
__global__ void host_wait_kernel(__nv_bfloat16* __restrict__ out, int H, HostResult* res, const unsigned long long* state) {
  // out: [ntok, Hh] flattened (H = ntok * Hh): the result rows follow the same layout
  const unsigned long long s = state[1];
  if (s == 0) return;
  __shared__ int ok;
  if (threadIdx.x == 0) {
    while (res->done != s) { __nanosleep(200); }
    __threadfence_system();
    ok = 1;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    const float v = __ldcv(&res->out[i]);
    out[i] = __float2bfloat16(__bfloat162float(out[i]) + v);
  }
}

std::tuple<torch::Tensor, torch::Tensor> offload_lookup(torch::Tensor ids, int64_t layer, int64_t E, torch::Tensor table,
                                                        torch::Tensor age, torch::Tensor clock, torch::Tensor w) {
  const int R = ids.numel();
  TORCH_CHECK(R <= 64);
  auto slots = torch::empty({R}, ids.options());
  auto gw = torch::empty({R}, w.options());
  offload_lookup_kernel<<<1, 64, 0, at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(), R, layer, E, table.data_ptr<int>(),
      age.data_ptr<int64_t>(), clock.data_ptr<int64_t>(), slots.data_ptr<int>(), gw.data_ptr<float>(), w.data_ptr<float>());
  return std::make_tuple(slots, gw);
}

void host_post(torch::Tensor ids, torch::Tensor w, int64_t layer, int64_t E, torch::Tensor table, torch::Tensor x,
               int64_t mailbox, torch::Tensor state) {
  host_post_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(), nullptr, w.data_ptr<float>(),
      ids.numel(), layer, E, table.data_ptr<int>(), (const __nv_bfloat16*)x.data_ptr(), x.size(-1), x.numel() / x.size(-1), (HostMailbox*)mailbox,
      (unsigned long long*)state.data_ptr<int64_t>());
}

torch::Tensor host_wait(torch::Tensor out, int64_t result, torch::Tensor state) {
  auto o = out.clone();
  host_wait_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>((__nv_bfloat16*)o.data_ptr(), o.numel(), (HostResult*)result,
      (const unsigned long long*)state.data_ptr<int64_t>());
  return o;
}

std::tuple<int64_t, int64_t, int64_t> host_tier_start(std::vector<int64_t> banks, int64_t H, int64_t I, int64_t G, std::vector<int64_t> cpus);
void host_tier_stop(int64_t handle);
int64_t host_tier_calls(int64_t handle);
std::vector<int64_t> host_tier_timing(int64_t handle);

// ---- host tier + background pool refresh ------------------------------------------------------------------------
// Lookup with a per-token clock: layer 0 advances it; hits get age = token clock.
__global__ void offload_lookup2_kernel(const int* __restrict__ ids, int R, int layer, int E, const int* __restrict__ table,
                                       long* __restrict__ age, long* __restrict__ clock, int* __restrict__ slots,
                                       float* __restrict__ gpu_w, const float* __restrict__ w) {
  __shared__ long now;
  if (threadIdx.x == 0) { if (layer == 0) clock[0] += 1; now = clock[0]; }
  __syncthreads();
  const int r = threadIdx.x;
  if (r >= R) return;
  const int s = table[layer * E + ids[r]];
  slots[r] = s >= 0 ? s : 0;
  gpu_w[r] = s >= 0 ? w[r] : 0.f;
  if (s >= 0) age[s] = now;
}

// Background refresh plan for layer `layer`'s misses: victims only among slots that are free, or hold an expert of a
// layer <= `layer` not used this token (age < token clock): nothing a later layer of this token reads is overwritten.
__global__ void refresh_plan_kernel(const int* __restrict__ ids, int R, int layer, int E, int* __restrict__ table,
                                    int* __restrict__ owner, long* __restrict__ age, const long* __restrict__ clock, int S,
                                    int max_installs, int* __restrict__ slots, int* __restrict__ miss) {
  __shared__ long best_age[32];
  __shared__ int best_slot[32];
  __shared__ int route_slot[32];
  __shared__ int installs;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarps = blockDim.x >> 5;
  const long now = clock[0];
  if (tid < R) route_slot[tid] = table[layer * E + ids[tid]];
  if (tid == 0) installs = 0;
  __syncthreads();
  for (int r = 0; r < R; ++r) {
    if (route_slot[r] >= 0 || installs >= max_installs) { if (tid == 0 && route_slot[r] < 0) route_slot[r] = -2; continue; }
    long a = 0x7fffffffffffffffL; int sl = -1;
    for (int i = tid; i < S; i += blockDim.x) {
      const int o = owner[i];
      const bool ok = o < 0 || (o / E <= layer && age[i] < now);
      const long v = o < 0 ? -1 : age[i];
      if (ok && v < a) { a = v; sl = i; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      long oa = __shfl_xor_sync(0xffffffffu, a, o); int os = __shfl_xor_sync(0xffffffffu, sl, o);
      if (oa < a || (oa == a && os >= 0 && (sl < 0 || os < sl))) { a = oa; sl = os; }
    }
    if (lane == 0) { best_age[warp] = a; best_slot[warp] = sl; }
    __syncthreads();
    if (tid == 0) {
      long ba = best_age[0]; int bs = best_slot[0];
      for (int w = 1; w < nwarps; ++w) if (best_slot[w] >= 0 && (bs < 0 || best_age[w] < ba)) { ba = best_age[w]; bs = best_slot[w]; }
      if (bs >= 0) {
        const int old = owner[bs];
        if (old >= 0) table[old] = -1;
        const int key = layer * E + ids[r];
        owner[bs] = key; table[key] = bs; age[bs] = now;
        route_slot[r] = bs | 0x40000000;
        ++installs;
      } else route_slot[r] = -2;
    }
    __syncthreads();
  }
  if (tid < R) {
    const int v = route_slot[tid];
    slots[tid] = v >= 0 ? (v & 0x3fffffff) : 0;
    miss[tid] = v >= 0 ? ((v >> 30) & 1) : 0;
  }
}

std::tuple<torch::Tensor, torch::Tensor> offload_lookup2(torch::Tensor ids, int64_t layer, int64_t E, torch::Tensor table,
                                                         torch::Tensor age, torch::Tensor clock, torch::Tensor w) {
  const int R = ids.numel();
  TORCH_CHECK(R <= 32);
  auto slots = torch::empty({R}, ids.options());
  auto gw = torch::empty({R}, w.options());
  offload_lookup2_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(), R, layer, E, table.data_ptr<int>(),
      age.data_ptr<int64_t>(), clock.data_ptr<int64_t>(), slots.data_ptr<int>(), gw.data_ptr<float>(), w.data_ptr<float>());
  return std::make_tuple(slots, gw);
}

std::tuple<torch::Tensor, torch::Tensor> refresh_plan(torch::Tensor ids, int64_t layer, int64_t E, torch::Tensor table,
                                                      torch::Tensor owner, torch::Tensor age, torch::Tensor clock,
                                                      int64_t max_installs) {
  const int R = ids.numel();
  TORCH_CHECK(R <= 32);
  auto slots = torch::empty({R}, ids.options()), miss = torch::empty({R}, ids.options());
  refresh_plan_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(), R, layer, E, table.data_ptr<int>(),
      owner.data_ptr<int>(), age.data_ptr<int64_t>(), clock.data_ptr<int64_t>(), owner.numel(), max_installs,
      slots.data_ptr<int>(), miss.data_ptr<int>());
  return std::make_tuple(slots, miss);
}

// ---- persistent MoE layer (host-tier mode): one cooperative launch per layer ----------------------------------
__device__ __forceinline__ void grid_barrier(unsigned int* bar, unsigned int nblocks) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(bar, 1u);
    while (atomicAdd(bar, 0u) < nblocks) { __nanosleep(32); }
    __threadfence();
  }
  __syncthreads();
}

// dot(x[0:K] bf16 (in global), dequant(row)) by one warp; K multiple of 1024 not required (chunks of 32 codes)
__device__ __forceinline__ float warp_row_dot(const uint4* __restrict__ wrow, const __nv_bfloat16* __restrict__ srow,
                                              const __nv_bfloat16* __restrict__ brow, const __nv_bfloat16* __restrict__ x,
                                              int K, int G, int lane) {
  float acc = 0.f;
  const int CH = K / 32;
  for (int c = lane; c < CH; c += 32) {
    const uint4 q = __ldg(wrow + c);
    const uint4* xp = reinterpret_cast<const uint4*>(x) + c * 4;
    float xf[32];
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      uint4 xv = xp[v];
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
      for (int p = 0; p < 4; ++p) { float2 f = __bfloat1622float2(h[p]); xf[v * 8 + 2 * p] = f.x; xf[v * 8 + 2 * p + 1] = f.y; }
    }
    float sx = 0.f;
#pragma unroll
    for (int q2 = 0; q2 < 32; ++q2) sx += xf[q2];
    const uint32_t words[4] = {q.x, q.y, q.z, q.w};
    float cx = 0.f;
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
      for (int t = 0; t < 8; ++t) cx = fmaf(code_f(words[u], t), xf[u * 8 + t], cx);
    const int g = (c * 32) / G;
    acc = fmaf(__bfloat162float(__ldg(srow + g)), cx, fmaf(__bfloat162float(__ldg(brow + g)), sx, acc));
  }
  return warp_sum(acc);
}

struct MoeLayerArgs {
  const __nv_bfloat16* x; const int* ids; const float* w; int R; int layer; int E;
  const int* table; long* age; const long* clock;
  const uint4* gw; const __nv_bfloat16* gs; const __nv_bfloat16* gb;   // pool gate/up banks [S, 2I, ...]
  const uint4* dw; const __nv_bfloat16* ds; const __nv_bfloat16* db;   // pool down banks [S, H, ...]
  int H, I, G;
  HostMailbox* mb; HostResult* res; unsigned long long* state;         // host tier (mb == nullptr: none)
  int* slots; float* gpu_w; __nv_bfloat16* gate_up; __nv_bfloat16* act;  // scratch [R], [R], [R, 2I], [R, I]
  unsigned int* bars; int* host_ready;                                  // per-call zeroed
  __nv_bfloat16* out;                                                   // [H]
  bool post_inside;                                                     // false: host_post ran before
};

__global__ void __launch_bounds__(256) moe_layer_host_kernel(MoeLayerArgs a) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gwarp = blockIdx.x * 8 + warp, nwarps = gridDim.x * 8;
  // phase 0: lookup + post (block 0)
  if (blockIdx.x == 0) {
    __shared__ int n;
    if (threadIdx.x == 0) n = 0;
    __syncthreads();
    if (threadIdx.x < a.R) {
      const int s = a.table[a.layer * a.E + a.ids[threadIdx.x]];
      a.slots[threadIdx.x] = s >= 0 ? s : 0;
      a.gpu_w[threadIdx.x] = s >= 0 ? a.w[threadIdx.x] : 0.f;
      if (s >= 0) a.age[s] = *a.clock;
    }
    __syncthreads();
    if (a.mb != nullptr && a.post_inside) {
      if (threadIdx.x == 0)
        for (int r = 0; r < a.R; ++r)
          if (a.table[a.layer * a.E + a.ids[r]] < 0) { a.mb->ids[n] = a.ids[r]; a.mb->w[n] = a.w[r]; a.mb->tok[n] = 0; ++n; }
      __syncthreads();
      if (n > 0) {
        const unsigned short* xs = reinterpret_cast<const unsigned short*>(a.x);
        for (int i = threadIdx.x; i < a.H; i += blockDim.x) a.mb->x[i] = xs[i];
        __syncthreads();
        if (threadIdx.x == 0) {
          a.mb->layer = a.layer; a.mb->n = n; a.mb->ntok = 1;
          __threadfence_system();
          const unsigned long long sq = a.state[0] + 1;
          a.state[0] = sq; a.state[1] = sq;
          a.mb->seq = sq;
          __threadfence_system();
        }
      } else if (threadIdx.x == 0) a.state[1] = 0;
    }
  }
  grid_barrier(a.bars + 0, gridDim.x);
  const int GU = 2 * a.I;
  // L2 prefetch of the resident routes' down banks (read in phase 2), spread over every thread, while phase 1 runs
  {
    const long line_vecs = 8;  // 128 bytes = 8 uint4
    const long per_expert = (long)a.H * (a.I / 32);  // uint4 per expert down bank
    const long lines = per_expert / line_vecs;
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x, nth = (long)gridDim.x * blockDim.x;
    for (int r = 0; r < a.R; ++r) {
      if (a.gpu_w[r] == 0.f) continue;
      const uint4* base = a.dw + (long)a.slots[r] * per_expert;
      for (long l = tid; l < lines; l += nth) asm volatile("prefetch.global.L2 [%0];" :: "l"(base + l * line_vecs));
    }
  }
  // phase 1: gate/up of the resident routes: a warp owns 4 consecutive rows of one route, lanes over 32-code chunks
  {
    const int CH = a.H / 32, groups = GU / 4;
    for (int t = gwarp; t < a.R * groups; t += nwarps) {
      const int r = t / groups, n0 = (t - r * groups) * 4;
      if (a.gpu_w[r] == 0.f) continue;
      const long e = a.slots[r];
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int c = lane; c < CH; c += 32) {
        uint4 qv[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) qv[k] = __ldg(a.gw + (e * GU + n0 + k) * CH + c);
        const uint4* xp = reinterpret_cast<const uint4*>(a.x) + c * 4;
        float xf[32];
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          uint4 xv = __ldg(xp + v);
          const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
          for (int p = 0; p < 4; ++p) { float2 f = __bfloat1622float2(h[p]); xf[v * 8 + 2 * p] = f.x; xf[v * 8 + 2 * p + 1] = f.y; }
        }
        float sx = 0.f;
#pragma unroll
        for (int q = 0; q < 32; ++q) sx += xf[q];
        const int g = (c * 32) / a.G;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const uint32_t words[4] = {qv[k].x, qv[k].y, qv[k].z, qv[k].w};
          float cx = 0.f;
#pragma unroll
          for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int tt = 0; tt < 8; ++tt) cx = fmaf(code_f(words[u], tt), xf[u * 8 + tt], cx);
          const long row = e * GU + n0 + k;
          acc[k] = fmaf(__bfloat162float(__ldg(a.gs + row * (a.H / a.G) + g)), cx,
                        fmaf(__bfloat162float(__ldg(a.gb + row * (a.H / a.G) + g)), sx, acc[k]));
        }
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float v = warp_sum(acc[k]);
        if (lane == 0) a.gate_up[(long)r * GU + n0 + k] = __float2bfloat16(v);
      }
    }
  }
  grid_barrier(a.bars + 1, gridDim.x);
  // phase 1b: act = bf16(bf16(silu(g)) * u), once, into the second half of the gate_up scratch rows' gate slots
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < (long)a.R * a.I; i += (long)gridDim.x * blockDim.x) {
    const int r = i / a.I, j = i - (long)r * a.I;
    const float g = __bfloat162float(a.gate_up[(long)r * GU + j]), u = __bfloat162float(a.gate_up[(long)r * GU + a.I + j]);
    const float sg = __bfloat162float(__float2bfloat16(g / (1.f + __expf(-g))));
    a.act[i] = __float2bfloat16(sg * u);
  }
  grid_barrier(a.bars + 3, gridDim.x);
  // phase 2: down rows, lanes over the flattened (route, 32-code chunk) space, routing-weighted sum
  {
    const int CH = a.I / 32;
    for (int n = gwarp; n < a.H; n += nwarps) {
      float acc = 0.f;
      for (int i = lane; i < a.R * CH; i += 32) {
        const int r = i / CH, c = i - r * CH;
        const float rw = a.gpu_w[r];
        if (rw == 0.f) continue;
        const long row = (long)a.slots[r] * a.H + n;
        const uint4 q = __ldg(a.dw + row * CH + c);
        const uint4* xp = reinterpret_cast<const uint4*>(a.act + (long)r * a.I) + c * 4;
        float xf[32];
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          uint4 xv = xp[v];
          const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
          for (int p = 0; p < 4; ++p) { float2 f = __bfloat1622float2(h[p]); xf[v * 8 + 2 * p] = f.x; xf[v * 8 + 2 * p + 1] = f.y; }
        }
        float sx = 0.f;
#pragma unroll
        for (int qq = 0; qq < 32; ++qq) sx += xf[qq];
        const uint32_t words[4] = {q.x, q.y, q.z, q.w};
        float cx = 0.f;
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
          for (int tt = 0; tt < 8; ++tt) cx = fmaf(code_f(words[u], tt), xf[u * 8 + tt], cx);
        const int g = (c * 32) / a.G;
        acc = fmaf(rw, fmaf(__bfloat162float(__ldg(a.ds + row * (a.I / a.G) + g)), cx,
                            __bfloat162float(__ldg(a.db + row * (a.I / a.G) + g)) * sx), acc);
      }
      acc = warp_sum(acc);
      if (lane == 0) a.out[n] = __float2bfloat16(acc);
    }
  }
  // phase 3: the host result (if posted): block 0 waits on the mapped flag, every warp adds it to its rows
  if (a.mb != nullptr) {
    const unsigned long long sq = a.state[1];
    if (sq != 0) {
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        while (a.res->done != sq) { __nanosleep(100); }
        __threadfence_system();
        atomicExch(a.host_ready, 1);
      }
      if (threadIdx.x == 0) { while (atomicAdd(a.host_ready, 0) == 0) { __nanosleep(64); } __threadfence(); }
      __syncthreads();
      for (int n = gwarp; n < a.H; n += nwarps)
        if (lane == 0) a.out[n] = __float2bfloat16(__bfloat162float(a.out[n]) + __ldcv(&a.res->out[n]));
    }
  }
}

torch::Tensor moe_layer_host(torch::Tensor x, torch::Tensor ids, torch::Tensor w, int64_t layer, int64_t E,
                             torch::Tensor table, torch::Tensor age, torch::Tensor clock,
                             std::vector<torch::Tensor> pool, int64_t mailbox, int64_t result, torch::Tensor state,
                             int64_t G, int64_t blocks, bool post_inside) {
  const int R = ids.numel(), H = x.numel(), I = pool[3].size(2) * 8;
  TORCH_CHECK(R <= 32 && G == 64 && x.scalar_type() == at::kBFloat16 && x.is_contiguous());
  auto opts = x.options();
  auto out = torch::empty({1, H}, opts);
  auto slots = torch::empty({R}, ids.options());
  auto gw = torch::empty({R}, w.options());
  auto gate_up = torch::empty({R, 2 * I}, opts);
  auto act = torch::empty({R, I}, opts);
  auto bars = torch::zeros({4}, ids.options());  // [0, 1, 3] barriers, [2] host-ready
  MoeLayerArgs a;
  a.x = (const __nv_bfloat16*)x.data_ptr(); a.ids = ids.data_ptr<int>(); a.w = w.data_ptr<float>(); a.R = R;
  a.layer = layer; a.E = E; a.table = table.data_ptr<int>(); a.age = age.data_ptr<int64_t>(); a.clock = clock.data_ptr<int64_t>();
  a.gw = (const uint4*)pool[0].data_ptr(); a.gs = (const __nv_bfloat16*)pool[1].data_ptr(); a.gb = (const __nv_bfloat16*)pool[2].data_ptr();
  a.dw = (const uint4*)pool[3].data_ptr(); a.ds = (const __nv_bfloat16*)pool[4].data_ptr(); a.db = (const __nv_bfloat16*)pool[5].data_ptr();
  a.H = H; a.I = I; a.G = G;
  a.mb = (HostMailbox*)mailbox; a.res = (HostResult*)result; a.state = (unsigned long long*)state.data_ptr<int64_t>();
  a.slots = slots.data_ptr<int>(); a.gpu_w = gw.data_ptr<float>(); a.gate_up = (__nv_bfloat16*)gate_up.data_ptr(); a.act = (__nv_bfloat16*)act.data_ptr();
  a.bars = (unsigned int*)bars.data_ptr<int>(); a.host_ready = bars.data_ptr<int>() + 2; a.out = (__nv_bfloat16*)out.data_ptr(); a.post_inside = post_inside;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(blocks); cfg.blockDim = dim3(256); cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeCooperative; attr[0].val.cooperative = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, moe_layer_host_kernel, a) == cudaSuccess, "moe_layer_host launch failed");
  return out;
}


// ---- v3 resident expert matvec: x / act staged in shared memory with per-chunk sums, weight loads issued ahead ------
// out[o, n] = sum_{j<J} rw(r) * dot(x[r / x_div], dequant(w[ids[r], n])),  r = o * J + j.  Block = 8 warps x ROWS rows.
namespace {
template <int G, int ROWS, int U>
__global__ void __launch_bounds__(256) qmv4_v3_kernel(
    const __nv_bfloat16* __restrict__ x, const uint4* __restrict__ w, const __nv_bfloat16* __restrict__ s,
    const __nv_bfloat16* __restrict__ b, const int* __restrict__ ids, const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ out, int N, int K, int J, int x_div, long w_se, long w_sn, long s_se, long s_sn,
    long b_se, long b_sn) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int CH = K / 32, o = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint4* xs = reinterpret_cast<uint4*>(smem);                       // [J][CH][4] bf16 chunks
  float* cs = reinterpret_cast<float*>(smem + (size_t)J * K * 2);   // [J][CH] chunk sums
  long* es = reinterpret_cast<long*>(cs + J * CH + (J * CH & 1));   // [J] expert ids
  float* ws = reinterpret_cast<float*>(es + J);                     // [J] route weights
  if (threadIdx.x < J) {
    const int r = o * J + threadIdx.x;
    es[threadIdx.x] = __ldg(ids + r);
    ws[threadIdx.x] = weights ? __ldg(weights + r) : 1.f;
  }
  for (int i = threadIdx.x; i < J * CH; i += 256) {
    const int j = i / CH, c = i - j * CH, r = o * J + j;
    const uint4* xp = reinterpret_cast<const uint4*>(x + (long)(r / x_div) * K) + c * 4;
    float sx = 0.f;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      uint4 xv = __ldg(xp + v);
      xs[i * 4 + v] = xv;
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
      for (int p = 0; p < 4; ++p) { float2 f = __bfloat1622float2(h[p]); sx += f.x + f.y; }
    }
    cs[i] = sx;
  }
  __syncthreads();
  const int n0 = (blockIdx.x * 8 + warp) * ROWS;
  if (n0 >= N) return;
  float acc[ROWS];
#pragma unroll
  for (int k = 0; k < ROWS; ++k) acc[k] = 0.f;
  const int items = J * CH;
  for (int base = lane; base < items; base += 32 * U) {
    uint4 qv[U][ROWS];
    int it[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      it[u] = base + 32 * u;
      if (it[u] < items) {
        const int j = it[u] / CH, c = it[u] - j * CH;
        const long e = es[j];
#pragma unroll
        for (int k = 0; k < ROWS; ++k) qv[u][k] = __ldg(w + (e * w_se + (long)(n0 + k) * w_sn) / 4 + c);
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (it[u] >= items) break;
      const int j = it[u] / CH, c = it[u] - j * CH;
      const long e = es[j];
      const float rw = ws[j], sx = cs[it[u]];
      const int g = (c * 32) / G;
      float xf[32];
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        uint4 xv = xs[it[u] * 4 + v];
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
        for (int p = 0; p < 4; ++p) { float2 f = __bfloat1622float2(h[p]); xf[v * 8 + 2 * p] = f.x; xf[v * 8 + 2 * p + 1] = f.y; }
      }
#pragma unroll
      for (int k = 0; k < ROWS; ++k) {
        const uint32_t words[4] = {qv[u][k].x, qv[u][k].y, qv[u][k].z, qv[u][k].w};
        float cx = 0.f;
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int t = 0; t < 8; ++t) cx = fmaf(code_f(words[q], t), xf[q * 8 + t], cx);
        const long row = n0 + k;
        const float sc = __bfloat162float(__ldg(s + e * s_se + row * s_sn + g));
        const float bi = __bfloat162float(__ldg(b + e * b_se + row * b_sn + g));
        acc[k] = fmaf(rw, fmaf(sc, cx, bi * sx), acc[k]);
      }
    }
  }
#pragma unroll
  for (int k = 0; k < ROWS; ++k) {
    float v = warp_sum(acc[k]);
    if (lane == 0) out[(long)o * N + n0 + k] = __float2bfloat16(v);
  }
}

template <int ROWS, int U>
void launch_v3(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s, const torch::Tensor& b,
               const torch::Tensor& ids, const float* weights, torch::Tensor& out, int64_t O, int64_t J, int64_t x_div,
               int64_t G) {
  const int64_t K = x.size(-1), N = w.size(1), CH = K / 32;
  TORCH_CHECK(N % (ROWS * 8) == 0 && w.stride(1) % 4 == 0 && w.stride(0) % 4 == 0, "rows / 16-byte alignment");
  const size_t sm = J * K * 2 + (J * CH + 1) * 4 + J * 8 + J * 4;
  dim3 grid(N / (ROWS * 8), O);
  auto stream = at::cuda::getCurrentCUDAStream();
#define L3(GS) do { auto kf = qmv4_v3_kernel<GS, ROWS, U>; \
      if (sm > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); \
      kf<<<grid, 256, sm, stream>>>( \
      (const __nv_bfloat16*)x.data_ptr(), (const uint4*)w.data_ptr(), (const __nv_bfloat16*)s.data_ptr(), \
      (const __nv_bfloat16*)b.data_ptr(), ids.data_ptr<int>(), weights, (__nv_bfloat16*)out.data_ptr(), N, K, J, x_div, \
      w.stride(0), w.stride(1), s.stride(0), s.stride(1), b.stride(0), b.stride(1)); } while (0)
  if (G == 32) L3(32); else if (G == 64) L3(64); else L3(128);
#undef L3
}
}  // namespace

torch::Tensor gather_qmv4_v3(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor b, torch::Tensor ids,
                             int64_t x_div, int64_t G, int64_t rows) {
  const int64_t K = x.size(-1), N = w.size(1), R = ids.numel();
  check(w, s, b, K, G);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && ids.scalar_type() == at::kInt && ids.is_contiguous());
  auto out = torch::empty({R, N}, x.options());
  if (!R) return out;
  if (rows == 1) launch_v3<1, 4>(x, w, s, b, ids, nullptr, out, R, 1, x_div, G);
  else if (rows == 2) launch_v3<2, 4>(x, w, s, b, ids, nullptr, out, R, 1, x_div, G);
  else launch_v3<4, 2>(x, w, s, b, ids, nullptr, out, R, 1, x_div, G);
  return out;
}

torch::Tensor down_route4_v3(torch::Tensor act, torch::Tensor w, torch::Tensor s, torch::Tensor b, torch::Tensor ids,
                             torch::Tensor weights, int64_t topk, int64_t G, int64_t rows) {
  const int64_t K = act.size(-1), N = w.size(1), R = ids.numel(), T = R / topk;
  check(w, s, b, K, G);
  TORCH_CHECK(act.scalar_type() == at::kBFloat16 && act.is_contiguous() && weights.scalar_type() == at::kFloat && weights.is_contiguous());
  auto out = torch::empty({T, N}, act.options());
  if (!T) return out;
  if (rows == 1) launch_v3<1, 4>(act, w, s, b, ids, weights.data_ptr<float>(), out, T, topk, 1, G);
  else if (rows == 2) launch_v3<2, 4>(act, w, s, b, ids, weights.data_ptr<float>(), out, T, topk, 1, G);
  else launch_v3<4, 2>(act, w, s, b, ids, weights.data_ptr<float>(), out, T, topk, 1, G);
  return out;
}


// ---- tensor-core expert matvec (resident banks, group size 64) -------------------------------------------------------
// Same contract as qmv4_kernel: out[o, n] = sum_{j<J} rw(r) * dot(x[r / x_div], dequant(w[ids[r], n])), r = o * J + j.
// A warp owns 16 rows; per (route, 64-code group) item it runs 4 mma.m16n8k16 (bf16 in, fp32 accumulate) with the codes
// dequantized two at a time as bf16 (128 + c) (one shift + one lop3 per pair), the activation in column 0 of B (the
// other columns zero). Per item: dot = scale * (D - 128 * sum_g(x)) + bias * sum_g(x). Lane q = lane % 4 owns 16 codes
// (one 8-byte load) of rows lane / 4 and lane / 4 + 8 per group; the activation is staged in shared memory with each
// 8-element block stored as the bf16 pairs (0,4) (1,5) (2,6) (3,7) the code pairs come out as.
namespace {
__device__ __forceinline__ uint32_t code_pair(uint32_t w, int k) {  // bf16x2 (128 + nibble k, 128 + nibble k + 4)
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(w >> (4 * k)), "r"(0x000F000Fu), "r"(0x43004300u));
  return r;
}
__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int RT, int KS, int U>
__global__ void __launch_bounds__(256) qmv4_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const uint2* __restrict__ w, const __nv_bfloat16* __restrict__ s,
    const __nv_bfloat16* __restrict__ b, const int* __restrict__ ids, const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ out, int N, int K, int J, int x_div, long w_se, long w_sn, long s_se, long s_sn,
    long b_se, long b_sn) {
  constexpr int G = 64;
  extern __shared__ __align__(16) unsigned char smem[];
  const int GPR = K / G, o = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint4* xs = reinterpret_cast<uint4*>(smem);                        // [J][K / 8] permuted 8-element blocks
  float* gs = reinterpret_cast<float*>(smem + (size_t)J * K * 2);    // [J][GPR] group sums
  float* red = gs + J * GPR;                                         // [KS][RT][16] split-K partials
  int* es = reinterpret_cast<int*>(red + KS * RT * 16);              // [J]
  float* ws = reinterpret_cast<float*>(es + J);                      // [J]
  if (threadIdx.x < J) {
    const int r = o * J + threadIdx.x;
    es[threadIdx.x] = __ldg(ids + r);
    ws[threadIdx.x] = weights ? __ldg(weights + r) : 1.f;
  }
  for (int i = threadIdx.x; i < J * K / 8; i += 256) {
    const int j = i / (K / 8), c = i - j * (K / 8), r = o * J + j;
    uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + (long)(r / x_div) * K) + c);
    const uint16_t* h = reinterpret_cast<const uint16_t*>(&xv);
    uint4 pv;
    pv.x = h[0] | ((uint32_t)h[4] << 16); pv.y = h[1] | ((uint32_t)h[5] << 16);
    pv.z = h[2] | ((uint32_t)h[6] << 16); pv.w = h[3] | ((uint32_t)h[7] << 16);
    xs[i] = pv;
  }
  for (int i = warp; i < J * GPR; i += 8) {  // group sums: a warp per (route, group), two elements per lane
    const int j = i / GPR, g = i - j * GPR, r = o * J + j;
    const __nv_bfloat162 v = reinterpret_cast<const __nv_bfloat162*>(x + (long)(r / x_div) * K + g * G)[lane];
    const float2 f = __bfloat1622float2(v);
    const float t = warp_sum(f.x + f.y);
    if (lane == 0) gs[i] = t;
  }
  __syncthreads();
  const int rt = warp % RT, ks = warp / RT;
  const int n0 = (blockIdx.x * RT + rt) * 16;
  const int q = lane & 3, rlo = n0 + (lane >> 2), rhi = rlo + 8;
  const bool col0 = lane < 4;  // B column 0 = the activation; D column 0 lives in lanes q == 0 (c0: row lo, c2: row hi)
  const int items = J * GPR;
  float acc_lo = 0.f, acc_hi = 0.f;
  for (int base = ks; base < items; base += KS * U) {
    uint2 wlo[U], whi[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int it = base + KS * u;
      if (it < items) {
        const int j = it / GPR, g = it - j * GPR;
        const long e = es[j];
        const long off = e * w_se / 2 + g * 4 + q;  // uint2 units: 64 codes = 32 bytes = 4 uint2 per (row, group)
        wlo[u] = __ldg(w + off + (long)rlo * w_sn / 2);
        whi[u] = __ldg(w + off + (long)rhi * w_sn / 2);
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int it = base + KS * u;
      if (it >= items) break;
      const int j = it / GPR, g = it - j * GPR;
      float d[4] = {0.f, 0.f, 0.f, 0.f};
      const uint32_t lo[2] = {wlo[u].x, wlo[u].y}, hi[2] = {whi[u].x, whi[u].y};
#pragma unroll
      for (int h = 0; h < 2; ++h) {  // word h of the lane's 16 codes: 8 codes -> two k16 steps
        uint4 xv = col0 ? xs[j * (K / 8) + g * 8 + q * 2 + h] : make_uint4(0, 0, 0, 0);
        mma_bf16(d, code_pair(lo[h], 0), code_pair(hi[h], 0), code_pair(lo[h], 1), code_pair(hi[h], 1), xv.x, xv.y);
        mma_bf16(d, code_pair(lo[h], 2), code_pair(hi[h], 2), code_pair(lo[h], 3), code_pair(hi[h], 3), xv.z, xv.w);
      }
      if (q == 0) {
        const long e = es[j];
        const float sg = gs[it], rw = ws[j];
        const float slo = __bfloat162float(__ldg(s + e * s_se + rlo * s_sn + g)), blo = __bfloat162float(__ldg(b + e * b_se + rlo * b_sn + g));
        const float shi = __bfloat162float(__ldg(s + e * s_se + rhi * s_sn + g)), bhi = __bfloat162float(__ldg(b + e * b_se + rhi * b_sn + g));
        acc_lo = fmaf(rw, fmaf(slo, d[0] - 128.f * sg, blo * sg), acc_lo);
        acc_hi = fmaf(rw, fmaf(shi, d[2] - 128.f * sg, bhi * sg), acc_hi);
      }
    }
  }
  if (KS == 1) {
    if (q == 0) { out[(long)o * N + rlo] = __float2bfloat16(acc_lo); out[(long)o * N + rhi] = __float2bfloat16(acc_hi); }
    return;
  }
  if (q == 0) { red[(ks * RT + rt) * 16 + (lane >> 2)] = acc_lo; red[(ks * RT + rt) * 16 + (lane >> 2) + 8] = acc_hi; }
  __syncthreads();
  if (threadIdx.x < RT * 16) {
    float v = 0.f;
#pragma unroll
    for (int k = 0; k < KS; ++k) v += red[k * RT * 16 + threadIdx.x];
    out[(long)o * N + blockIdx.x * RT * 16 + threadIdx.x] = __float2bfloat16(v);
  }
}

template <int RT, int KS, int U>
void launch_mma(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s, const torch::Tensor& b,
                const torch::Tensor& ids, const float* weights, torch::Tensor& out, int64_t O, int64_t J, int64_t x_div) {
  const int64_t K = x.size(-1), N = w.size(1), GPR = K / 64;
  TORCH_CHECK(N % (RT * 16) == 0 && w.stride(1) % 2 == 0 && w.stride(0) % 2 == 0, "rows / 8-byte alignment");
  const size_t sm = J * K * 2 + J * GPR * 4 + KS * RT * 16 * 4 + J * 8;
  auto kf = qmv4_mma_kernel<RT, KS, U>;
  if (sm > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  kf<<<dim3(N / (RT * 16), O), 256, sm, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)x.data_ptr(), (const uint2*)w.data_ptr(), (const __nv_bfloat16*)s.data_ptr(),
      (const __nv_bfloat16*)b.data_ptr(), ids.data_ptr<int>(), weights, (__nv_bfloat16*)out.data_ptr(), N, K, J, x_div,
      w.stride(0), w.stride(1), s.stride(0), s.stride(1), b.stride(0), b.stride(1));
}

template <int U>
void mma_dispatch(int64_t rt, const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s, const torch::Tensor& b,
                  const torch::Tensor& ids, const float* weights, torch::Tensor& out, int64_t O, int64_t J, int64_t x_div) {
  switch (rt) {
    case 1: launch_mma<1, 8, U>(x, w, s, b, ids, weights, out, O, J, x_div); break;
    case 2: launch_mma<2, 4, U>(x, w, s, b, ids, weights, out, O, J, x_div); break;
    case 4: launch_mma<4, 2, U>(x, w, s, b, ids, weights, out, O, J, x_div); break;
    default: launch_mma<8, 1, U>(x, w, s, b, ids, weights, out, O, J, x_div); break;
  }
}
}  // namespace

torch::Tensor gather_qmv4_mma(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor b, torch::Tensor ids,
                              int64_t x_div, int64_t G, int64_t rt) {
  const int64_t K = x.size(-1), N = w.size(1), R = ids.numel();
  check(w, s, b, K, G);
  TORCH_CHECK(G == 64, "mma path: group size 64");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && ids.scalar_type() == at::kInt && ids.is_contiguous());
  auto out = torch::empty({R, N}, x.options());
  if (R) mma_dispatch<4>(rt, x, w, s, b, ids, nullptr, out, R, 1, x_div);
  return out;
}

torch::Tensor down_route4_mma(torch::Tensor act, torch::Tensor w, torch::Tensor s, torch::Tensor b, torch::Tensor ids,
                              torch::Tensor weights, int64_t topk, int64_t G, int64_t rt) {
  const int64_t K = act.size(-1), N = w.size(1), R = ids.numel(), T = R / topk;
  check(w, s, b, K, G);
  TORCH_CHECK(G == 64, "mma path: group size 64");
  TORCH_CHECK(act.scalar_type() == at::kBFloat16 && act.is_contiguous() && weights.scalar_type() == at::kFloat && weights.is_contiguous());
  auto out = torch::empty({T, N}, act.options());
  if (T) mma_dispatch<4>(rt, act, w, s, b, ids, weights.data_ptr<float>(), out, T, topk, 1);
  return out;
}


// ---- v5 tensor-core expert matvec: contiguous row tiles staged with cp.async ---------------------------------------
// Same contract as qmv4_kernel (group size 64). A block owns 16 output rows of output o: it copies, for each of its J
// routes, the 16 rows' codes (one contiguous span of the expert) and scales / biases into shared memory with 16-byte
// cp.async (coalesced, one round trip after the ids), then 8 warps split the (route, group) items and run
// mma.m16n8k16 with codes dequantized as bf16 (128 + c); dot = scale * (D - 128 * sum_g(x)) + bias * sum_g(x).
namespace {
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(gmem));
}
__device__ __forceinline__ float bsum8(uint4 v) {
  float t = 0.f;
  const uint32_t a[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) t += __uint_as_float(a[i] << 16) + __uint_as_float(a[i] & 0xFFFF0000u);
  return t;
}

template <int JMAX>
__global__ void __launch_bounds__(256) qmv4_tc_kernel(
    const __nv_bfloat16* __restrict__ x, const uint4* __restrict__ w, const __nv_bfloat16* __restrict__ s,
    const __nv_bfloat16* __restrict__ b, const int* __restrict__ ids, const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ out, int N, int K, int J, int x_div, long w_se, long s_se, long b_se) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int GPR = K / 64, RB = K / 2 + 16;  // row bytes in smem (padded: rows land 16 bytes apart in the banks)
  const int o = blockIdx.y, n0 = blockIdx.x * 16, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  unsigned char* ws_ = smem;                                                    // [J][16][RB]
  __nv_bfloat16* ss = reinterpret_cast<__nv_bfloat16*>(smem + J * 16 * RB);     // [J][16][GPR]
  __nv_bfloat16* bs = ss + J * 16 * GPR;                                        // [J][16][GPR]
  float* red = reinterpret_cast<float*>(bs + J * 16 * GPR + 8);                 // [8][16]
  uint4* xs = reinterpret_cast<uint4*>(red + 8 * 16);                           // [J][K / 8] the routes' activations
  int e_[JMAX];
#pragma unroll
  for (int j = 0; j < JMAX; ++j) e_[j] = j < J ? __ldg(ids + o * J + j) : 0;
  const int CPR = K / 32;  // 16-byte chunks per row
  for (int i = threadIdx.x; i < J * 16 * CPR; i += 256) {
    const int j = i / (16 * CPR), rem = i - j * 16 * CPR, r = rem / CPR, c = rem - r * CPR;
    int e = e_[0];
#pragma unroll
    for (int jj = 1; jj < JMAX; ++jj) if (jj == j) e = e_[jj];
    cp_async16(ws_ + (j * 16 + r) * RB + c * 16, w + (long)e * (w_se / 4) + (long)(n0 + r) * CPR + c);
  }
  const int SCH = 16 * GPR * 2 / 16;  // 16-byte chunks of one route's 16-row scale tile
  for (int i = threadIdx.x; i < 2 * J * SCH; i += 256) {
    const int which = i / (J * SCH), rem = i - which * J * SCH, j = rem / SCH, c = rem - j * SCH;
    int e = e_[0];
#pragma unroll
    for (int jj = 1; jj < JMAX; ++jj) if (jj == j) e = e_[jj];
    const __nv_bfloat16* src = (which ? b + e * b_se : s + e * s_se) + (long)n0 * GPR;
    cp_async16((which ? bs : ss) + j * 16 * GPR + c * 8, src + c * 8);
  }
  for (int i = threadIdx.x; i < J * K / 8; i += 256) {
    const int j = i / (K / 8), c = i - j * (K / 8);
    cp_async16(xs + i, reinterpret_cast<const uint4*>(x + (long)((o * J + j) / x_div) * K) + c);
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  const int q = lane & 3, rl = lane >> 2;
  const bool col0 = lane < 4;
  float acc_lo = 0.f, acc_hi = 0.f;
  for (int it = warp; it < J * GPR; it += 8) {
    const int j = it / GPR, g = it - j * GPR, r = o * J + j;
    const uint2 lo = *reinterpret_cast<const uint2*>(ws_ + (j * 16 + rl) * RB + g * 32 + q * 8);
    const uint2 hi = *reinterpret_cast<const uint2*>(ws_ + (j * 16 + rl + 8) * RB + g * 32 + q * 8);
    uint4 xv[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    if (col0) {
      const uint4* xp = xs + j * (K / 8) + g * 8 + q * 2;
      xv[0] = xp[0]; xv[1] = xp[1];
    }
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    const uint32_t L[2] = {lo.x, lo.y}, Hh[2] = {hi.x, hi.y};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t p0 = __byte_perm(xv[h].x, xv[h].z, 0x5410), p1 = __byte_perm(xv[h].x, xv[h].z, 0x7632);
      const uint32_t p2 = __byte_perm(xv[h].y, xv[h].w, 0x5410), p3 = __byte_perm(xv[h].y, xv[h].w, 0x7632);
      mma_bf16(d, code_pair(L[h], 0), code_pair(Hh[h], 0), code_pair(L[h], 1), code_pair(Hh[h], 1), p0, p1);
      mma_bf16(d, code_pair(L[h], 2), code_pair(Hh[h], 2), code_pair(L[h], 3), code_pair(Hh[h], 3), p2, p3);
    }
    float sg = bsum8(xv[0]) + bsum8(xv[1]);
    sg += __shfl_xor_sync(0xffffffffu, sg, 1);
    sg += __shfl_xor_sync(0xffffffffu, sg, 2);
    sg = __shfl_sync(0xffffffffu, sg, 0);
    if (q == 0) {
      const float rw = weights ? __ldg(weights + r) : 1.f;
      const int so = j * 16 * GPR + g;
      const float slo = __bfloat162float(ss[so + rl * GPR]), blo = __bfloat162float(bs[so + rl * GPR]);
      const float shi = __bfloat162float(ss[so + (rl + 8) * GPR]), bhi = __bfloat162float(bs[so + (rl + 8) * GPR]);
      acc_lo = fmaf(rw, fmaf(slo, d[0] - 128.f * sg, blo * sg), acc_lo);
      acc_hi = fmaf(rw, fmaf(shi, d[2] - 128.f * sg, bhi * sg), acc_hi);
    }
  }
  if (q == 0) { red[warp * 16 + rl] = acc_lo; red[warp * 16 + rl + 8] = acc_hi; }
  __syncthreads();
  if (threadIdx.x < 16) {
    float v = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) v += red[k * 16 + threadIdx.x];
    out[(long)o * N + n0 + threadIdx.x] = __float2bfloat16(v);
  }
}

void launch_tc(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s, const torch::Tensor& b,
               const torch::Tensor& ids, const float* weights, torch::Tensor& out, int64_t O, int64_t J, int64_t x_div) {
  const int64_t K = x.size(-1), N = w.size(1), GPR = K / 64;
  TORCH_CHECK(N % 16 == 0 && J <= 16, "N % 16, J <= 16");
  TORCH_CHECK(w.stride(1) * 8 == K && w.stride(0) == N * w.stride(1) && s.stride(1) == GPR && s.stride(0) == N * GPR &&
              b.stride(1) == GPR && b.stride(0) == N * GPR, "contiguous expert banks");
  const size_t sm = J * 16 * (K / 2 + 16) + 2 * J * 16 * GPR * 2 + 16 + 8 * 16 * 4 + J * K * 2;
  auto stream = at::cuda::getCurrentCUDAStream();
#define LT(JM) do { auto kf = qmv4_tc_kernel<JM>; \
    if (sm > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); \
    kf<<<dim3(N / 16, O), 256, sm, stream>>>((const __nv_bfloat16*)x.data_ptr(), (const uint4*)w.data_ptr(), \
      (const __nv_bfloat16*)s.data_ptr(), (const __nv_bfloat16*)b.data_ptr(), ids.data_ptr<int>(), weights, \
      (__nv_bfloat16*)out.data_ptr(), N, K, J, x_div, w.stride(0), s.stride(0), b.stride(0)); } while (0)
  if (J == 1) LT(1); else if (J <= 4) LT(4); else if (J <= 10) LT(10); else LT(16);
#undef LT
}
}  // namespace

torch::Tensor gather_qmv4_tc(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor b, torch::Tensor ids,
                             int64_t x_div, int64_t G) {
  const int64_t K = x.size(-1), N = w.size(1), R = ids.numel();
  check(w, s, b, K, G);
  TORCH_CHECK(G == 64 && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && ids.scalar_type() == at::kInt && ids.is_contiguous());
  auto out = torch::empty({R, N}, x.options());
  if (R) launch_tc(x, w, s, b, ids, nullptr, out, R, 1, x_div);
  return out;
}

torch::Tensor down_route4_tc(torch::Tensor act, torch::Tensor w, torch::Tensor s, torch::Tensor b, torch::Tensor ids,
                             torch::Tensor weights, int64_t topk, int64_t G) {
  const int64_t K = act.size(-1), N = w.size(1), R = ids.numel(), T = R / topk;
  check(w, s, b, K, G);
  TORCH_CHECK(G == 64 && act.scalar_type() == at::kBFloat16 && act.is_contiguous() && weights.scalar_type() == at::kFloat && weights.is_contiguous());
  auto out = torch::empty({T, N}, act.options());
  if (T) launch_tc(act, w, s, b, ids, weights.data_ptr<float>(), out, T, topk, 1);
  return out;
}


// ---- v6 tensor-core expert matvec (group size 64), resident or streamed ------------------------------------------
// out[o, n] = sum_{j<J} rw(r) * dot(x[r / x_div], dequant(W_r[n])), r = o * J + j, where W_r is pool slot
// (slots ? slots[r] : ids[r]), or, when miss[r], expert ids[r] of the host banks (UVA) -- then the block writes the tile
// it staged into that pool slot (install). Block = 16 output rows of output o for all J routes, 8 warps over the
// (route, group) items. One cp.async batch after the ids (codes, scales / biases, activations), then: activations
// permuted to bf16 pair order + group sums, then per item 4 mma.m16n8k16 with codes as bf16 (128 + c).
namespace {
namespace tc6 {
__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(smem)), "l"(gmem));
}
struct Banks { const uint4* w; const __nv_bfloat16* s; const __nv_bfloat16* b; };
template <int K, int JMAX>
__global__ void __launch_bounds__(256) kernel(
    const __nv_bfloat16* __restrict__ x, uint4* pw, __nv_bfloat16* ps, __nv_bfloat16* pb, Banks host,
    const int* __restrict__ ids, const int* __restrict__ slots, const int* __restrict__ miss,
    const float* __restrict__ weights, __nv_bfloat16* __restrict__ out, int N, int J, int x_div) {
  constexpr int GPR = K / 64, CPR = K / 32, RB = K / 2 + 16, ROWS = 16;
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* wsm = smem;                                                      // [J][16][RB]
  __nv_bfloat16* ssm = reinterpret_cast<__nv_bfloat16*>(smem + JMAX * ROWS * RB);  // [J][16][GPR]
  __nv_bfloat16* bsm = ssm + JMAX * ROWS * GPR;
  uint4* xsm = reinterpret_cast<uint4*>(bsm + JMAX * ROWS * GPR);                // [J][K / 8]
  float* gsm = reinterpret_cast<float*>(xsm + JMAX * K / 8);                      // [J][GPR]
  float* red = gsm + JMAX * GPR;                                                  // [8][16]
  const int o = blockIdx.y, n0 = blockIdx.x * ROWS, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const long ew = (long)N * CPR, es = (long)N * GPR;  // per-expert sizes (uint4 / bf16)
  const uint4* wsrc[JMAX];
  const __nv_bfloat16* ssrc[JMAX];
  const __nv_bfloat16* bsrc[JMAX];
  bool mis[JMAX];
  long dst[JMAX];
#pragma unroll
  for (int j = 0; j < JMAX; ++j) {
    if (j < J) {
      const int r = o * J + j;
      const long e = __ldg(ids + r), sl = slots ? (long)__ldg(slots + r) : e;
      mis[j] = miss && __ldg(miss + r);
      dst[j] = sl;
      wsrc[j] = (mis[j] ? host.w + e * ew : pw + sl * ew) + (long)n0 * CPR;
      ssrc[j] = (mis[j] ? host.s + e * es : ps + sl * es) + (long)n0 * GPR;
      bsrc[j] = (mis[j] ? host.b + e * es : pb + sl * es) + (long)n0 * GPR;
    }
  }
#pragma unroll
  for (int j = 0; j < JMAX; ++j) {
    if (j >= J) break;
    for (int i = tid; i < ROWS * CPR; i += 256) cp16(wsm + (j * ROWS + i / CPR) * RB + (i % CPR) * 16, wsrc[j] + i);
    constexpr int SC = ROWS * GPR / 8;
    for (int c = tid; c < 2 * SC; c += 256)
      cp16(c < SC ? ssm + j * ROWS * GPR + c * 8 : bsm + j * ROWS * GPR + (c - SC) * 8, c < SC ? ssrc[j] + c * 8 : bsrc[j] + (c - SC) * 8);
    const uint4* xr = reinterpret_cast<const uint4*>(x + (long)((o * J + j) / x_div) * K);
    for (int c = tid; c < K / 8; c += 256) cp16(xsm + j * (K / 8) + c, xr + c);
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
#pragma unroll
  for (int j = 0; j < JMAX; ++j) {  // install: the staged tiles of the missed routes into their pool slots
    if (j >= J) break;
    if (!mis[j]) continue;
    uint4* wd = pw + dst[j] * ew + (long)n0 * CPR;
    for (int i = tid; i < ROWS * CPR; i += 256) wd[i] = *reinterpret_cast<const uint4*>(wsm + (j * ROWS + i / CPR) * RB + (i % CPR) * 16);
    constexpr int SC = ROWS * GPR / 8;
    uint4* sd = reinterpret_cast<uint4*>(ps + dst[j] * es + (long)n0 * GPR);
    uint4* bd = reinterpret_cast<uint4*>(pb + dst[j] * es + (long)n0 * GPR);
    for (int c = tid; c < 2 * SC; c += 256)
      (c < SC ? sd[c] : bd[c - SC]) = reinterpret_cast<const uint4*>(c < SC ? ssm + j * ROWS * GPR : bsm + j * ROWS * GPR)[c < SC ? c : c - SC];
  }
  for (int i = tid; i < J * K / 8; i += 256) {  // activations -> pair order, group sums (8 chunks of 8 = a group)
    uint4 v = xsm[i], pv;
    pv.x = __byte_perm(v.x, v.z, 0x5410); pv.y = __byte_perm(v.x, v.z, 0x7632);
    pv.z = __byte_perm(v.y, v.w, 0x5410); pv.w = __byte_perm(v.y, v.w, 0x7632);
    xsm[i] = pv;
    float t = 0.f;
    const uint32_t a[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) t += __uint_as_float(a[q] << 16) + __uint_as_float(a[q] & 0xFFFF0000u);
    t += __shfl_xor_sync(0xffffffffu, t, 1);
    t += __shfl_xor_sync(0xffffffffu, t, 2);
    t += __shfl_xor_sync(0xffffffffu, t, 4);
    if ((tid & 7) == 0) gsm[i / 8] = t;
  }
  __syncthreads();
  const int q = lane & 3, rl = lane >> 2;
  const bool col0 = lane < 4;
  float acc_lo = 0.f, acc_hi = 0.f;
  for (int it = warp; it < J * GPR; it += 8) {
    const int j = it / GPR, g = it - j * GPR;
    const unsigned char* wr = wsm + (j * ROWS + rl) * RB + g * 32 + q * 8;
    const uint2 lo = *reinterpret_cast<const uint2*>(wr), hi = *reinterpret_cast<const uint2*>(wr + 8 * RB);
    uint4 xa = make_uint4(0, 0, 0, 0), xb = xa;
    if (col0) { const uint4* xp = xsm + j * (K / 8) + g * 8 + q * 2; xa = xp[0]; xb = xp[1]; }
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    mma_bf16(d, code_pair(lo.x, 0), code_pair(hi.x, 0), code_pair(lo.x, 1), code_pair(hi.x, 1), xa.x, xa.y);
    mma_bf16(d, code_pair(lo.x, 2), code_pair(hi.x, 2), code_pair(lo.x, 3), code_pair(hi.x, 3), xa.z, xa.w);
    mma_bf16(d, code_pair(lo.y, 0), code_pair(hi.y, 0), code_pair(lo.y, 1), code_pair(hi.y, 1), xb.x, xb.y);
    mma_bf16(d, code_pair(lo.y, 2), code_pair(hi.y, 2), code_pair(lo.y, 3), code_pair(hi.y, 3), xb.z, xb.w);
    if (q == 0) {
      const float sg = gsm[it], rw = weights ? __ldg(weights + o * J + j) : 1.f;
      const int so = (j * ROWS + rl) * GPR + g;
      acc_lo = fmaf(rw, fmaf(__bfloat162float(ssm[so]), d[0] - 128.f * sg, __bfloat162float(bsm[so]) * sg), acc_lo);
      acc_hi = fmaf(rw, fmaf(__bfloat162float(ssm[so + 8 * GPR]), d[2] - 128.f * sg, __bfloat162float(bsm[so + 8 * GPR]) * sg), acc_hi);
    }
  }
  if (q == 0) { red[warp * 16 + rl] = acc_lo; red[warp * 16 + rl + 8] = acc_hi; }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int w8 = 0; w8 < 8; ++w8) v += red[w8 * 16 + tid];
    out[(long)o * N + n0 + tid] = __float2bfloat16(v);
  }
}
template <int K, int JMAX>
void launch(const torch::Tensor& x, const torch::Tensor& pw, const torch::Tensor& ps, const torch::Tensor& pb, Banks host,
            const torch::Tensor& ids, const int* slots, const int* miss, const float* weights, torch::Tensor& out, int64_t O,
            int64_t J, int64_t x_div) {
  const int64_t N = pw.size(1);
  constexpr size_t sm = JMAX * 16 * (K / 2 + 16) + 2 * JMAX * 16 * (K / 64) * 2 + JMAX * K * 2 + JMAX * (K / 64) * 4 + 8 * 16 * 4;
  auto kf = kernel<K, JMAX>;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); attr = true; }
  kf<<<dim3(N / 16, O), 256, sm, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)x.data_ptr(), (uint4*)pw.data_ptr(), (__nv_bfloat16*)ps.data_ptr(), (__nv_bfloat16*)pb.data_ptr(),
      host, ids.data_ptr<int>(), slots, miss, weights, (__nv_bfloat16*)out.data_ptr(), N, J, x_div);
}
}  // namespace tc6
}  // namespace

bool qmv4_tc6_supported(int64_t K, int64_t J, int64_t G) {
  return G == 64 && ((K == 2560 && J == 1) || (K == 640 && J <= 10));
}

torch::Tensor qmv4_tc6(torch::Tensor x, torch::Tensor pw, torch::Tensor ps, torch::Tensor pb, int64_t hw, int64_t hs,
                       int64_t hb, torch::Tensor ids, std::optional<torch::Tensor> slots, std::optional<torch::Tensor> miss,
                       std::optional<torch::Tensor> weights, int64_t J, int64_t x_div, int64_t G) {
  const int64_t K = x.size(-1), N = pw.size(1), R = ids.numel(), O = R / J;
  TORCH_CHECK(qmv4_tc6_supported(K, J, G), "tc6: (K=2560, J=1) or (K=640, J<=10), group size 64");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && ids.scalar_type() == at::kInt && ids.is_contiguous());
  TORCH_CHECK(pw.is_contiguous() && ps.is_contiguous() && pb.is_contiguous() && N % 16 == 0 && pw.size(2) * 8 == K &&
              ps.size(2) * 64 == K && ps.scalar_type() == at::kBFloat16, "contiguous slot-indexed banks [S, N, ...]");
  TORCH_CHECK(!miss || (slots && hw), "streamed: slots + host banks");
  auto out = torch::empty({O, N}, x.options());
  if (!O) return out;
  tc6::Banks host{(const uint4*)hw, (const __nv_bfloat16*)hs, (const __nv_bfloat16*)hb};
  const int* sp = slots ? slots->data_ptr<int>() : nullptr;
  const int* mp = miss ? miss->data_ptr<int>() : nullptr;
  const float* wp = weights ? weights->data_ptr<float>() : nullptr;
  if (K == 2560) tc6::launch<2560, 1>(x, pw, ps, pb, host, ids, sp, mp, wp, out, O, J, x_div);
  else tc6::launch<640, 10>(x, pw, ps, pb, host, ids, sp, mp, wp, out, O, J, x_div);
  return out;
}


// ---- host tier v2: post / wait ---------------------------------------------------------------------------------
// Layout shared with host_tier2.cpp (Mailbox2 / Result2).
namespace {
constexpr int HT2_MAXR = 64, HT2_MAXT = 4, HT2_MAXH = 8192, HT2_RING = 1024;
struct alignas(64) HostMailbox2 {
  volatile unsigned long long seq; volatile unsigned long long hint_seq;
  int layer, n, ntok, hint_layer, hint_n, pad[5];
  int ids[HT2_MAXR]; float w[HT2_MAXR]; int tok[HT2_MAXR]; int hint_ids[HT2_MAXR];
  alignas(64) unsigned int xq[HT2_MAXT * HT2_MAXH / 2];
  alignas(64) float xgs[HT2_MAXT * (HT2_MAXH / 64)];
  alignas(64) volatile unsigned long long inst_head;
  alignas(64) volatile unsigned long long inst_done;
  alignas(64) int ring[HT2_RING][2];
};
struct alignas(64) HostResult2 { volatile unsigned long long done; unsigned long long pad[7]; alignas(64) float out[HT2_MAXT * HT2_MAXH]; };

// One block. Routes whose expert is not resident -> mailbox (compacted in route order); the activation rows with 16-byte
// stores; per-64-group sums (a warp per group). state[0]: sequence counter, state[1]: the sequence to wait for (0: none).
// A route is the CPU's iff the lookup gave it GPU weight 0 (`gpu_w`): the GPU and the CPU always agree on the split,
// even when a concurrent refresh commits an install between the lookup and this post.
__global__ void __launch_bounds__(512) host_post2_kernel(const int* __restrict__ ids, const float* __restrict__ w, int R,
    int layer, const float* __restrict__ gpu_w, const __nv_bfloat16* __restrict__ x, int H, int ntok,
    HostMailbox2* mb, unsigned long long* state) {
  __shared__ int miss_flag[HT2_MAXR];
  __shared__ int n;
  const int tid = threadIdx.x, topk = R / ntok;
  if (tid < R) miss_flag[tid] = gpu_w[tid] == 0.f && w[tid] != 0.f;
  __syncthreads();
  if (tid == 0) {
    int c = 0;
    for (int r = 0; r < R; ++r)
      if (miss_flag[r]) { mb->ids[c] = ids[r]; mb->w[c] = w[r]; mb->tok[c] = r / topk; ++c; }
    n = c;
  }
  __syncthreads();
  if (n == 0) { if (tid == 0) state[1] = 0; return; }
  // activation in the host tier's pair layout: per 128-element chunk c, word c*64 + k*16 + i = (x[c*128+8i+k], x[+4])
  const unsigned short* xs = reinterpret_cast<const unsigned short*>(x);
  uint4* xd = reinterpret_cast<uint4*>(mb->xq);
  for (int v = tid; v < ntok * H / 8; v += blockDim.x) {  // 4 words per store
    const int o = v * 4, row = o / (H / 2), ow = o - row * (H / 2), c = ow / 64, k = (ow % 64) / 16, i0 = ow % 16;
    const unsigned short* xr = xs + (long)row * H + c * 128;
    unsigned int wv[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) wv[u] = xr[8 * (i0 + u) + k] | ((unsigned int)xr[8 * (i0 + u) + k + 4] << 16);
    xd[v] = make_uint4(wv[0], wv[1], wv[2], wv[3]);
  }
  const int lane = tid & 31, warp = tid >> 5, nwarps = blockDim.x >> 5;
  for (int gi = warp; gi < ntok * H / 64; gi += nwarps) {
    const float2 f = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(x + (long)gi * 64)[lane]);
    float v = f.x + f.y;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) mb->xgs[gi] = v;
  }
  __syncthreads();
  if (tid == 0) {
    mb->layer = layer; mb->n = n; mb->ntok = ntok;
    __threadfence_system();
    const unsigned long long s = state[0] + 1;
    state[0] = s; state[1] = s;
    mb->seq = s;
    __threadfence_system();
  }
}

// out + host result: thread 0 spins on the done flag, then every thread reads 16 bytes of the result (one round).
__global__ void __launch_bounds__(1024) host_wait2_kernel(const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
    int numel, HostResult2* res, const unsigned long long* state) {
  const unsigned long long s = state[1];
  if (s != 0 && threadIdx.x == 0) {
    while (res->done != s) {}
    __threadfence_system();
  }
  __syncthreads();
  for (int i = threadIdx.x; i < numel / 4; i += blockDim.x) {
    float4 v = s != 0 ? __ldcv(reinterpret_cast<const float4*>(res->out) + i) : make_float4(0.f, 0.f, 0.f, 0.f);
    float2 a0 = make_float2(0.f, 0.f), a1 = a0;
    if (in) {
      const __nv_bfloat162* a = reinterpret_cast<const __nv_bfloat162*>(in) + 2 * i;
      a0 = __bfloat1622float2(a[0]); a1 = __bfloat1622float2(a[1]);
    }
    __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(out) + 2 * i;
    o[0] = __floats2bfloat162_rn(a0.x + v.x, a0.y + v.y);
    o[1] = __floats2bfloat162_rn(a1.x + v.z, a1.y + v.w);
  }
}

// Prefetch hint: the experts of `ids` (guesses for `layer`) that are not resident -> the host tier stages them.
__global__ void host_hint2_kernel(const int* __restrict__ ids, int R, int layer, int E, const int* __restrict__ table,
                                  HostMailbox2* mb, unsigned long long* hstate) {
  __shared__ int e_s[HT2_MAXR], keep[HT2_MAXR];
  const int r = threadIdx.x;
  if (r < R) { e_s[r] = ids[r]; keep[r] = table[layer * E + e_s[r]] < 0; }
  __syncthreads();
  if (r < R && keep[r])
    for (int q = 0; q < r; ++q) if (e_s[q] == e_s[r]) { keep[r] = 0; break; }  // first occurrence only
  __syncthreads();
  if (r != 0) return;
  int c = 0;
  for (int q = 0; q < R; ++q) if (keep[q]) mb->hint_ids[c++] = e_s[q];
  if (c == 0) return;
  mb->hint_layer = layer; mb->hint_n = c;
  __threadfence_system();
  const unsigned long long s = hstate[0] + 1;
  hstate[0] = s;
  mb->hint_seq = s;
  __threadfence_system();
}
}  // namespace

void host_hint2(torch::Tensor ids, int64_t layer, int64_t E, torch::Tensor table, int64_t mailbox, torch::Tensor hstate) {
  TORCH_CHECK(ids.numel() <= HT2_MAXR && ids.scalar_type() == at::kInt && ids.is_contiguous());
  host_hint2_kernel<<<1, 64, 0, at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(), ids.numel(), layer, E,
      table.data_ptr<int>(), (HostMailbox2*)mailbox, (unsigned long long*)hstate.data_ptr<int64_t>());
}
namespace {

// Guess + hint in one launch (side stream): logits = x . W[e] for the E router rows (warp per row, all blocks), the
// last block to finish takes the top-k, drops resident experts (table of `layer`) and posts them as the hint.
// scratch: [E] float logits; counter: one int (reset by the last block).
__global__ void __launch_bounds__(256) host_guess2_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ W,
    int E, int H, int k, int max_out, int layer, const int* __restrict__ table, float* scratch, unsigned int* counter,
    HostMailbox2* mb, unsigned long long* hstate) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gw = blockIdx.x * 8 + warp, nw = gridDim.x * 8;
  for (int e = gw; e < E; e += nw) {
    const uint4* wr = reinterpret_cast<const uint4*>(W + (long)e * H);
    const uint4* xr = reinterpret_cast<const uint4*>(x);
    float acc = 0.f;
    for (int c = lane; c < H / 8; c += 32) {
      const uint4 wv = __ldg(wr + c), xv = __ldg(xr + c);
      const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
      const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
      for (int p = 0; p < 4; ++p) { const float2 a = __bfloat1622float2(w2[p]), b = __bfloat1622float2(x2[p]); acc = fmaf(a.x, b.x, fmaf(a.y, b.y, acc)); }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) scratch[e] = acc;
  }
  __shared__ bool last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(counter, 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  __shared__ float bv[8];
  __shared__ int bi[8], picked[HT2_MAXR];
  __shared__ int n_out;
  if (threadIdx.x == 0) n_out = 0;
  for (int r = 0; r < k; ++r) {  // k rounds of argmax over E (taken entries set to -inf)
    float v = -INFINITY; int idx = -1;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
      float t = __ldcg(scratch + e);
      if (t != t) t = -INFINITY;  // NaN logits (garbage activations) never win and never yield an index < 0 below
      if (idx < 0 || t > v) { v = t; idx = e; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, v, o); const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
      if (oi >= 0 && (idx < 0 || ov > v || (ov == v && oi < idx))) { v = ov; idx = oi; }
    }
    if (lane == 0) { bv[warp] = v; bi[warp] = idx; }
    __syncthreads();
    if (threadIdx.x == 0) {
      float b = bv[0]; int ix = bi[0];
      for (int w = 1; w < 8; ++w) if (bi[w] >= 0 && (ix < 0 || bv[w] > b || (bv[w] == b && bi[w] < ix))) { b = bv[w]; ix = bi[w]; }
      if (ix >= 0) {
        scratch[ix] = -INFINITY;
        if (table[layer * E + ix] < 0 && n_out < max_out) picked[n_out++] = ix;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *counter = 0;
    if (n_out) {
      for (int q = 0; q < n_out; ++q) mb->hint_ids[q] = picked[q];
      mb->hint_layer = layer; mb->hint_n = n_out;
      __threadfence_system();
      const unsigned long long s = hstate[0] + 1;
      hstate[0] = s;
      mb->hint_seq = s;
      __threadfence_system();
    }
  }
}
}  // namespace

void host_guess2(torch::Tensor x, torch::Tensor W, int64_t k, int64_t layer, torch::Tensor table, int64_t mailbox,
                 torch::Tensor hstate, torch::Tensor scratch, torch::Tensor counter, int64_t max_out, int64_t blocks) {
  const int64_t E = W.size(0), H = W.size(1);
  TORCH_CHECK(x.numel() == H && x.scalar_type() == at::kBFloat16 && W.scalar_type() == at::kBFloat16 && W.is_contiguous() &&
              x.is_contiguous() && H % 8 == 0 && k <= HT2_MAXR && scratch.numel() >= E && counter.scalar_type() == at::kInt);
  host_guess2_kernel<<<(int)blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)x.data_ptr(),
      (const __nv_bfloat16*)W.data_ptr(), E, H, k, (int)std::min<int64_t>(max_out, HT2_MAXR), layer, table.data_ptr<int>(), scratch.data_ptr<float>(),
      (unsigned int*)counter.data_ptr<int>(), (HostMailbox2*)mailbox, (unsigned long long*)hstate.data_ptr<int64_t>());
}
namespace {

// Pool refresh for the host tier (side stream, after the layer's post): (1) commit the installs the CPU installer has
// copied (table[key] = slot unless the slot was re-assigned since), (2) plan installs for this layer's missed experts:
// LRU victim among slots of layers <= `layer` not used this token (no kernel can be reading them), its old key
// invalidated now, the new key marked pending (-2: a miss for every lookup) until committed. rstate: [produced,
// committed]. Planning stops at `max_installs` or when the ring is full.
__global__ void __launch_bounds__(1024) host_refresh2_kernel(const int* __restrict__ ids, int R, int layer, int E,
    int* __restrict__ table, int* __restrict__ owner, long* __restrict__ age, const long* __restrict__ clock, int S,
    int max_installs, HostMailbox2* mb, long long* rstate) {
  __shared__ long best_age[32];
  __shared__ int best_slot[32];
  __shared__ int keys[HT2_MAXR];
  __shared__ int nk;
  __shared__ long long head;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarps = blockDim.x >> 5;
  const long now = clock[0];
  if (tid == 0) {
    const long long done = (long long)mb->inst_done;
    __threadfence_system();
    for (long long i = rstate[1]; i < done; ++i) {
      const int key = mb->ring[i % HT2_RING][0], slot = mb->ring[i % HT2_RING][1];
      if (owner[slot] == key && table[key] == -2) table[key] = slot;
    }
    rstate[1] = done;
    head = rstate[0];
    // this layer's missed, not pending, distinct experts
    int c = 0;
    for (int r = 0; r < R && c < max_installs; ++r) {
      const int key = layer * E + ids[r];
      if (table[key] != -1) continue;
      bool dup = false;
      for (int q = 0; q < c; ++q) dup |= keys[q] == key;
      if (!dup) keys[c++] = key;
    }
    nk = (head - done + c <= HT2_RING) ? c : 0;
  }
  __syncthreads();
  for (int q = 0; q < nk; ++q) {
    long a = 0x7fffffffffffffffL; int sl = -1;
    for (int i = tid; i < S; i += blockDim.x) {
      const int o = owner[i];
      const bool ok = o < 0 || (o / E <= layer && age[i] < now);
      const long v = o < 0 ? -1 : age[i];
      if (ok && v < a) { a = v; sl = i; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const long oa = __shfl_xor_sync(0xffffffffu, a, o); const int os = __shfl_xor_sync(0xffffffffu, sl, o);
      if (os >= 0 && (sl < 0 || oa < a || (oa == a && os < sl))) { a = oa; sl = os; }
    }
    if (lane == 0) { best_age[warp] = a; best_slot[warp] = sl; }
    __syncthreads();
    if (tid == 0) {
      long ba = best_age[0]; int bs = best_slot[0];
      for (int w = 1; w < nwarps; ++w) if (best_slot[w] >= 0 && (bs < 0 || best_age[w] < ba)) { ba = best_age[w]; bs = best_slot[w]; }
      if (bs >= 0) {
        const int old = owner[bs];
        if (old >= 0) table[old] = -1;
        owner[bs] = keys[q]; table[keys[q]] = -2; age[bs] = now;
        mb->ring[head % HT2_RING][0] = keys[q]; mb->ring[head % HT2_RING][1] = bs;
        ++head;
      }
    }
    __syncthreads();
  }
  if (tid == 0 && head != rstate[0]) {
    rstate[0] = head;
    __threadfence_system();
    mb->inst_head = (unsigned long long)head;
    __threadfence_system();
  }
}
}  // namespace

void host_refresh2(torch::Tensor ids, int64_t layer, int64_t E, torch::Tensor table, torch::Tensor owner, torch::Tensor age,
                   torch::Tensor clock, int64_t max_installs, int64_t mailbox, torch::Tensor rstate) {
  TORCH_CHECK(ids.numel() <= HT2_MAXR && max_installs <= HT2_MAXR && rstate.scalar_type() == at::kLong && rstate.numel() == 2);
  host_refresh2_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(), ids.numel(), layer, E,
      table.data_ptr<int>(), owner.data_ptr<int>(), age.data_ptr<int64_t>(), clock.data_ptr<int64_t>(), owner.numel(),
      max_installs, (HostMailbox2*)mailbox, (long long*)rstate.data_ptr<int64_t>());
}
void host_tier2_installer(int64_t handle, std::vector<int64_t> pool, std::vector<int64_t> expert_bytes, int64_t E, int64_t device, int64_t cpu);
int64_t host_tier2_drain(int64_t handle);
namespace {

// ---- host tier v3: no fork / join on the critical path -----------------------------------------------------------
// lookup + post in one launch on the main stream: slots / GPU weights for the resident routes (LRU ages, per-token clock
// at layer 0), the others posted to the CPU workers (ids, weights, token, activation in pair layout, group sums).
// state[0]: post counter, state[1]: sequence this layer waits for (0: nothing posted).
__global__ void __launch_bounds__(512) host_lookup_post3_kernel(const int* __restrict__ ids, const float* __restrict__ w,
    int R, int layer, int E, const int* __restrict__ table, long* __restrict__ age, long* __restrict__ clock,
    int* __restrict__ slots, float* __restrict__ gpu_w, const __nv_bfloat16* __restrict__ x, int H, int ntok,
    HostMailbox2* mb, unsigned long long* state) {
  __shared__ int miss_flag[HT2_MAXR];
  __shared__ int n;
  __shared__ long now;
  const int tid = threadIdx.x, topk = R / ntok;
  if (tid == 0) { if (layer == 0) clock[0] += 1; now = clock[0]; }
  __syncthreads();
  if (tid < R) {
    const int sl = table[layer * E + ids[tid]];
    slots[tid] = sl >= 0 ? sl : 0;
    gpu_w[tid] = sl >= 0 ? w[tid] : 0.f;
    if (sl >= 0) age[sl] = now;
    miss_flag[tid] = sl < 0 && w[tid] != 0.f;
  }
  __syncthreads();
  if (tid == 0) {
    int c = 0;
    for (int r = 0; r < R; ++r)
      if (miss_flag[r]) { mb->ids[c] = ids[r]; mb->w[c] = w[r]; mb->tok[c] = r / topk; ++c; }
    n = c;
  }
  __syncthreads();
  if (n == 0) { if (tid == 0) state[1] = 0; return; }
  const unsigned short* xs = reinterpret_cast<const unsigned short*>(x);
  uint4* xd = reinterpret_cast<uint4*>(mb->xq);
  for (int v = tid; v < ntok * H / 8; v += blockDim.x) {
    const int o = v * 4, row = o / (H / 2), ow = o - row * (H / 2), c = ow / 64, k = (ow % 64) / 16, i0 = ow % 16;
    const unsigned short* xr = xs + (long)row * H + c * 128;
    unsigned int wv[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) wv[u] = xr[8 * (i0 + u) + k] | ((unsigned int)xr[8 * (i0 + u) + k + 4] << 16);
    xd[v] = make_uint4(wv[0], wv[1], wv[2], wv[3]);
  }
  const int lane = tid & 31, warp = tid >> 5, nwarps = blockDim.x >> 5;
  for (int gi = warp; gi < ntok * H / 64; gi += nwarps) {
    const float2 f = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(x + (long)gi * 64)[lane]);
    float v = f.x + f.y;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) mb->xgs[gi] = v;
  }
  __syncthreads();
  if (tid == 0) {
    mb->layer = layer; mb->n = n; mb->ntok = ntok;
    __threadfence_system();
    const unsigned long long sq = state[0] + 1;
    state[0] = sq; state[1] = sq;
    mb->seq = sq;
  }
}

// out + the CPU result of this layer's post (out as is when nothing was posted): spins on the mapped done flag.
__global__ void __launch_bounds__(1024) host_join3_kernel(const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
    int numel, HostResult2* res, const unsigned long long* state) {
  const unsigned long long sq = state[1];
  if (sq != 0 && threadIdx.x == 0) {
    while (res->done != sq) {}
    __threadfence_system();
  }
  __syncthreads();
  for (int i = threadIdx.x; i < numel / 4; i += blockDim.x) {
    const float4 v = sq != 0 ? __ldcv(reinterpret_cast<const float4*>(res->out) + i) : make_float4(0.f, 0.f, 0.f, 0.f);
    const __nv_bfloat162* a = reinterpret_cast<const __nv_bfloat162*>(in) + 2 * i;
    const float2 a0 = __bfloat1622float2(a[0]), a1 = __bfloat1622float2(a[1]);
    __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(out) + 2 * i;
    o[0] = __floats2bfloat162_rn(a0.x + v.x, a0.y + v.y);
    o[1] = __floats2bfloat162_rn(a1.x + v.z, a1.y + v.w);
  }
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor> host_lookup_post3(torch::Tensor ids, torch::Tensor w, int64_t layer, int64_t E,
    torch::Tensor table, torch::Tensor age, torch::Tensor clock, torch::Tensor x, int64_t mailbox, torch::Tensor state) {
  const int64_t H = x.size(-1), ntok = x.numel() / H, R = ids.numel();
  TORCH_CHECK(R <= HT2_MAXR && ntok <= HT2_MAXT && H <= HT2_MAXH && H % 128 == 0 && x.is_contiguous() &&
              x.scalar_type() == at::kBFloat16 && ids.scalar_type() == at::kInt && w.scalar_type() == at::kFloat);
  auto slots = torch::empty({R}, ids.options());
  auto gw = torch::empty({R}, w.options());
  host_lookup_post3_kernel<<<1, 512, 0, at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(), w.data_ptr<float>(), R,
      layer, E, table.data_ptr<int>(), age.data_ptr<int64_t>(), clock.data_ptr<int64_t>(), slots.data_ptr<int>(),
      gw.data_ptr<float>(), (const __nv_bfloat16*)x.data_ptr(), H, ntok, (HostMailbox2*)mailbox,
      (unsigned long long*)state.data_ptr<int64_t>());
  return std::make_tuple(slots, gw);
}

torch::Tensor host_join3(torch::Tensor out, int64_t result, torch::Tensor state) {
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.numel() % 4 == 0 && out.numel() <= HT2_MAXT * HT2_MAXH);
  auto o = torch::empty_like(out);
  host_join3_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)out.data_ptr(),
      (__nv_bfloat16*)o.data_ptr(), out.numel(), (HostResult2*)result, (const unsigned long long*)state.data_ptr<int64_t>());
  return o;
}
namespace {

// ---- next-layer MoE input estimate (hyper-connection mix of the streams entering the layer, no attention) ---------
// xhat[i] = mean_g sigmoid(up @ silu(down @ hn / G))[g*H+i] * hn[g*H+i],  hn = grouped RMSNorm(streams) * (1 + w)
// scratch: float [G*H (hn) + R (d)], R = down rows (low rank). For the host tier's hints (approximate, fp32).
__global__ void hc_norm_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, float* hn, int H, float eps) {
  // one block per group
  const int g = blockIdx.x;
  __shared__ float red[32];
  float ss = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) { const float v = __bfloat162float(x[g * H + i]); ss += v * v; }
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    if (threadIdx.x == 0) red[0] = rsqrtf(t / H + eps);
  }
  __syncthreads();
  const float r = red[0];
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    hn[g * H + i] = __bfloat162float(x[g * H + i]) * r * (1.f + __bfloat162float(w[g * H + i]));
}
// out[row] = act(dot(W[row], v)) for row < N (warp per row), K-long rows, v fp32
template <int ACT>  // 0: silu(z / scale), 1: sigmoid
__global__ void rowdot_kernel(const __nv_bfloat16* __restrict__ W, const float* __restrict__ v, float* out, int N, int K, float scale) {
  const int lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= N) return;
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(W + (long)row * K);
  float acc = 0.f;
  for (int c = lane; c < K / 2; c += 32) { const float2 a = __bfloat1622float2(w2[c]); acc = fmaf(a.x, v[2 * c], fmaf(a.y, v[2 * c + 1], acc)); }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) {
    if (ACT == 0) { const float z = acc / scale; out[row] = z / (1.f + __expf(-z)); }
    else out[row] = 1.f / (1.f + __expf(-acc));
  }
}
__global__ void hc_mix_kernel(const float* __restrict__ u, const float* __restrict__ hn, __nv_bfloat16* xhat, int H, int G) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= H) return;
  float acc = 0.f;
  for (int g = 0; g < G; ++g) acc += u[g * H + i] * hn[g * H + i];
  xhat[i] = __float2bfloat16(acc / G);
}
}  // namespace

void hc_mix_guess(torch::Tensor streams, torch::Tensor norm_w, torch::Tensor down, torch::Tensor up, double eps,
                  torch::Tensor xhat, torch::Tensor scratch) {
  const int64_t H = xhat.numel(), GH = streams.numel(), G = GH / H, R = down.size(0);
  TORCH_CHECK(GH == G * H && down.size(1) == GH && up.size(0) == GH && up.size(1) == R && norm_w.numel() == GH &&
              scratch.numel() >= 2 * GH + R && streams.is_contiguous() && down.is_contiguous() && up.is_contiguous() &&
              streams.scalar_type() == at::kBFloat16 && down.scalar_type() == at::kBFloat16 && norm_w.scalar_type() == at::kBFloat16);
  auto st = at::cuda::getCurrentCUDAStream();
  float* hn = scratch.data_ptr<float>();
  float* d = hn + GH;
  float* u = d + R;
  hc_norm_kernel<<<G, 1024, 0, st>>>((const __nv_bfloat16*)streams.data_ptr(), (const __nv_bfloat16*)norm_w.data_ptr(), hn, H, (float)eps);
  rowdot_kernel<0><<<(R + 7) / 8, 256, 0, st>>>((const __nv_bfloat16*)down.data_ptr(), hn, d, R, GH, (float)G);
  rowdot_kernel<1><<<(GH + 7) / 8, 256, 0, st>>>((const __nv_bfloat16*)up.data_ptr(), d, u, GH, R, 1.f);
  hc_mix_kernel<<<(H + 255) / 256, 256, 0, st>>>(u, hn, (__nv_bfloat16*)xhat.data_ptr(), H, G);
}
namespace {
}  // namespace

void host_post2(torch::Tensor ids, torch::Tensor w, int64_t layer, torch::Tensor gpu_w, torch::Tensor x,
                int64_t mailbox, torch::Tensor state) {
  const int64_t H = x.size(-1), ntok = x.numel() / H;
  TORCH_CHECK(ids.numel() <= HT2_MAXR && ntok <= HT2_MAXT && H <= HT2_MAXH && H % 128 == 0 && x.is_contiguous() &&
              x.scalar_type() == at::kBFloat16 && ids.scalar_type() == at::kInt && w.scalar_type() == at::kFloat &&
              gpu_w.scalar_type() == at::kFloat && gpu_w.numel() == ids.numel());
  host_post2_kernel<<<1, 512, 0, at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(), w.data_ptr<float>(), ids.numel(),
      layer, gpu_w.data_ptr<float>(), (const __nv_bfloat16*)x.data_ptr(), H, ntok, (HostMailbox2*)mailbox,
      (unsigned long long*)state.data_ptr<int64_t>());
}

torch::Tensor host_wait2(torch::Tensor out, int64_t result, torch::Tensor state) {
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.numel() % 4 == 0 && out.numel() <= HT2_MAXT * HT2_MAXH);
  auto o = torch::empty_like(out);
  host_wait2_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)out.data_ptr(),
      (__nv_bfloat16*)o.data_ptr(), out.numel(), (HostResult2*)result, (const unsigned long long*)state.data_ptr<int64_t>());
  return o;
}

// dst[:numel] = host result of the last post (0 when nothing was posted): for a side stream, into a static buffer
void host_fetch2(torch::Tensor dst, int64_t numel, int64_t result, torch::Tensor state) {
  TORCH_CHECK(dst.is_contiguous() && dst.scalar_type() == at::kBFloat16 && numel % 4 == 0 && numel <= dst.numel() && numel <= HT2_MAXT * HT2_MAXH);
  host_wait2_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(nullptr, (__nv_bfloat16*)dst.data_ptr(), numel,
      (HostResult2*)result, (const unsigned long long*)state.data_ptr<int64_t>());
}

std::tuple<int64_t, int64_t, int64_t> host_tier2_start(std::vector<int64_t> banks, int64_t H, int64_t I, int64_t G, std::vector<int64_t> cpus, int64_t E);
void host_tier2_stop(int64_t handle);
std::vector<int64_t> host_tier2_timing(int64_t handle);


// ---- diagnostics: GPU global timer (ns) into buf[*slot % n], then ++*slot (one thread; captured in CUDA graphs) ----
__global__ void gtimer_kernel(long long* buf, long long* slot, int n) {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  const long long i = *slot;
  buf[i % n] = (long long)t;
  *slot = i + 1;
}
void gtimer(torch::Tensor buf, torch::Tensor slot) {
  gtimer_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>((long long*)buf.data_ptr<int64_t>(), (long long*)slot.data_ptr<int64_t>(), buf.numel());
}

TORCH_LIBRARY(OPS_NS, m) {
  m.def("gtimer(Tensor(a!) buf, Tensor(b!) slot) -> ()");
  m.def("host_hint2(Tensor ids, int layer, int E, Tensor table, int mailbox, Tensor(a!) hstate) -> ()");
  m.def("host_fetch2(Tensor(a!) dst, int numel, int result, Tensor state) -> ()");
  m.def("host_guess2(Tensor x, Tensor W, int k, int layer, Tensor table, int mailbox, Tensor(a!) hstate, Tensor(b!) scratch, Tensor(c!) counter, int max_out, int blocks) -> ()");
  m.def("host_refresh2(Tensor ids, int layer, int E, Tensor(a!) table, Tensor(b!) owner, Tensor(c!) age, Tensor clock, int max_installs, int mailbox, Tensor(d!) rstate) -> ()");
  m.def("host_tier2_installer(int handle, int[] pool, int[] expert_bytes, int E, int device, int cpu) -> ()");
  m.def("host_tier2_drain(int handle) -> int");
  m.def("host_lookup_post3(Tensor ids, Tensor w, int layer, int E, Tensor table, Tensor age, Tensor clock, Tensor x, int mailbox, Tensor state) -> (Tensor, Tensor)");
  m.def("host_join3(Tensor out, int result, Tensor state) -> Tensor");
  m.def("hc_mix_guess(Tensor streams, Tensor norm_w, Tensor down, Tensor up, float eps, Tensor(a!) xhat, Tensor(b!) scratch) -> ()");
  m.def("host_post2(Tensor ids, Tensor w, int layer, Tensor gpu_w, Tensor x, int mailbox, Tensor(a!) state) -> ()");
  m.def("host_wait2(Tensor out, int result, Tensor state) -> Tensor");
  m.def("host_tier2_start(int[] banks, int H, int I, int G, int[] cpus, int E) -> (int, int, int)");
  m.def("host_tier2_stop(int handle) -> ()");
  m.def("host_tier2_timing(int handle) -> int[]");
  m.def("qmv4_tc6(Tensor x, Tensor pw, Tensor ps, Tensor pb, int hw, int hs, int hb, Tensor ids, Tensor? slots, Tensor? miss, Tensor? weights, int J, int x_div, int group_size) -> Tensor");
  m.def("gather_qmv4_tc(Tensor x, Tensor w, Tensor s, Tensor b, Tensor ids, int x_div, int group_size) -> Tensor");
  m.def("down_route4_tc(Tensor act, Tensor w, Tensor s, Tensor b, Tensor ids, Tensor weights, int top_k, int group_size) -> Tensor");
  m.def("gather_qmv4_mma(Tensor x, Tensor w, Tensor s, Tensor b, Tensor ids, int x_div, int group_size, int rt) -> Tensor");
  m.def("down_route4_mma(Tensor act, Tensor w, Tensor s, Tensor b, Tensor ids, Tensor weights, int top_k, int group_size, int rt) -> Tensor");
  m.def("gather_qmv4_v3(Tensor x, Tensor w, Tensor s, Tensor b, Tensor ids, int x_div, int group_size, int rows) -> Tensor");
  m.def("down_route4_v3(Tensor act, Tensor w, Tensor s, Tensor b, Tensor ids, Tensor weights, int top_k, int group_size, int rows) -> Tensor");
  m.def("moe_layer_host(Tensor x, Tensor ids, Tensor w, int layer, int E, Tensor table, Tensor(a!) age, Tensor clock, Tensor[] pool, int mailbox, int result, Tensor(b!) state, int group_size, int blocks, bool post_inside) -> Tensor");
  m.def("offload_lookup2(Tensor ids, int layer, int E, Tensor table, Tensor(a!) age, Tensor(b!) clock, Tensor w) -> (Tensor, Tensor)");
  m.def("refresh_plan(Tensor ids, int layer, int E, Tensor(a!) table, Tensor(b!) owner, Tensor(c!) age, Tensor clock, int max_installs) -> (Tensor, Tensor)");
  m.def("offload_lookup(Tensor ids, int layer, int E, Tensor table, Tensor(a!) age, Tensor clock, Tensor w) -> (Tensor, Tensor)");
  m.def("host_post(Tensor ids, Tensor w, int layer, int E, Tensor table, Tensor x, int mailbox, Tensor(a!) state) -> ()");
  m.def("host_wait(Tensor out, int result, Tensor state) -> Tensor");
  m.def("host_tier_start(int[] banks, int H, int I, int G, int[] cpus) -> (int, int, int)");
  m.def("host_tier_stop(int handle) -> ()");
  m.def("host_tier_calls(int handle) -> int");
  m.def("host_tier_timing(int handle) -> int[]");
  m.def("qmv4_offload_v2(Tensor x, Tensor pw, Tensor ps, Tensor pb, int hw, int hs, int hb, Tensor ids, Tensor? slots, Tensor? miss, Tensor? weights, int J, int x_div, int group_size, int rows) -> Tensor");
  m.def("router_topk(Tensor x, Tensor W, int k, bool norm) -> (Tensor, Tensor, Tensor)");
  m.def("offload_install(Tensor ids, Tensor slots, Tensor miss, int[] host, Tensor[] pool, int blocks) -> ()");
  m.def("offload_plan(Tensor ids, int layer, int E, Tensor(a!) table, Tensor(b!) owner, Tensor(c!) age, Tensor(d!) clock) -> (Tensor, Tensor)");
  m.def("qmv4_offload(Tensor x, Tensor pw, Tensor ps, Tensor pb, int hw, int hs, int hb, Tensor ids, Tensor slots, Tensor miss, Tensor? weights, int J, int x_div, int group_size) -> Tensor");
  m.def("gemv_bf16(Tensor x, Tensor W, Tensor? bias) -> Tensor");
  m.def("gather_qmv4(Tensor x, Tensor w, Tensor s, Tensor b, Tensor ids, int x_div, int group_size) -> Tensor");
  m.def("down_route4(Tensor act, Tensor w, Tensor s, Tensor b, Tensor ids, Tensor weights, int top_k, int group_size) -> Tensor");
}
TORCH_LIBRARY_IMPL(OPS_NS, CUDA, m) {
  m.impl("gtimer", &gtimer);
  m.impl("host_post2", &host_post2);
  m.impl("hc_mix_guess", &hc_mix_guess);
  m.impl("host_lookup_post3", &host_lookup_post3);
  m.impl("host_join3", &host_join3);
  m.impl("host_refresh2", &host_refresh2);
  m.impl("host_guess2", &host_guess2);
  m.impl("host_fetch2", &host_fetch2);
  m.impl("host_hint2", &host_hint2);
  m.impl("host_wait2", &host_wait2);
  m.impl("qmv4_tc6", &qmv4_tc6);
  m.impl("gather_qmv4_tc", &gather_qmv4_tc);
  m.impl("down_route4_tc", &down_route4_tc);
  m.impl("gather_qmv4_mma", &gather_qmv4_mma);
  m.impl("down_route4_mma", &down_route4_mma);
  m.impl("gather_qmv4_v3", &gather_qmv4_v3);
  m.impl("down_route4_v3", &down_route4_v3);
  m.impl("gather_qmv4", &gather_qmv4);
  m.impl("down_route4", &down_route4);
  m.impl("gemv_bf16", &gemv_bf16);
  m.impl("offload_plan", &offload_plan);
  m.impl("qmv4_offload", &qmv4_offload);
  m.impl("offload_install", &offload_install);
  m.impl("router_topk", &router_topk);
  m.impl("qmv4_offload_v2", &qmv4_offload_v2);
  m.impl("offload_lookup", &offload_lookup);
  m.impl("host_post", &host_post);
  m.impl("host_wait", &host_wait);
  m.impl("offload_lookup2", &offload_lookup2);
  m.impl("refresh_plan", &refresh_plan);
  m.impl("moe_layer_host", &moe_layer_host);
}
TORCH_LIBRARY_IMPL(OPS_NS, CompositeExplicitAutograd, m) {
  m.impl("host_tier2_start", &host_tier2_start);
  m.impl("host_tier2_installer", &host_tier2_installer);
  m.impl("host_tier2_drain", &host_tier2_drain);
  m.impl("host_tier2_stop", &host_tier2_stop);
  m.impl("host_tier2_timing", &host_tier2_timing);
  m.impl("host_tier_start", &host_tier_start);
  m.impl("host_tier_stop", &host_tier_stop);
  m.impl("host_tier_calls", &host_tier_calls);
  m.impl("host_tier_timing", &host_tier_timing);
}

// The Python module the kernels loader imports (ops are registered by the static TORCH_LIBRARY blocks above)
#include <Python.h>
#define _CONCAT(a, b) a##b
#define CONCAT(a, b) _CONCAT(a, b)
#define _STR(a) #a
#define STR(a) _STR(a)
extern "C" PyObject* CONCAT(PyInit_, OPS_NS)(void) {
  static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STR(OPS_NS), nullptr, 0, nullptr};
  return PyModule_Create(&module);
}
