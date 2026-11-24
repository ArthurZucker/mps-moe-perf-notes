// Batch-small MoE expert matvecs on the MetalExperts affine 4-bit layout (codes LSB-first in uint32 words,
// per-group bf16 scales / biases), for sm_100. One warp per output row; coalesced word loads; x read 16 bytes at a time.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace {

constexpr int WARPS = 8;

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
        for (int t = 0; t < 8; ++t) cx = fmaf(float((words[u] >> (4 * t)) & 0xF), xf[u * 8 + t], cx);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_qmv4", &gather_qmv4);
  m.def("down_route4", &down_route4);
}
