#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstdlib>
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
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

template <int RT, int KS, int U, int MODE>
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
  if (!(MODE & 4)) {
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
  } else if (threadIdx.x < J) { es[threadIdx.x] = o; ws[threadIdx.x] = 1.f; }
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
        wlo[u] = (MODE & 2) ? make_uint2(it, e) : __ldg(w + off + (long)rlo * w_sn / 2);
        whi[u] = (MODE & 2) ? make_uint2(e, it) : __ldg(w + off + (long)rhi * w_sn / 2);
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
        if (!(MODE & 8)) {
        mma_bf16(d, code_pair(lo[h], 0), code_pair(hi[h], 0), code_pair(lo[h], 1), code_pair(hi[h], 1), xv.x, xv.y);
        mma_bf16(d, code_pair(lo[h], 2), code_pair(hi[h], 2), code_pair(lo[h], 3), code_pair(hi[h], 3), xv.z, xv.w);
        } else { d[0] += __uint_as_float(lo[h] ^ hi[h] ^ xv.x); }
      }
      if (q == 0) {
        const long e = es[j];
        const float sg = gs[it], rw = ws[j];
        const float slo = (MODE & 1) ? 1.f : __bfloat162float(__ldg(s + e * s_se + rlo * s_sn + g)), blo = (MODE & 1) ? 0.5f : __bfloat162float(__ldg(b + e * b_se + rlo * b_sn + g));
        const float shi = (MODE & 1) ? 1.f : __bfloat162float(__ldg(s + e * s_se + rhi * s_sn + g)), bhi = (MODE & 1) ? 0.5f : __bfloat162float(__ldg(b + e * b_se + rhi * b_sn + g));
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


template <int MODE>
float bench(const __nv_bfloat16* x, const uint2* w, const __nv_bfloat16* s, const __nv_bfloat16* b, int* ids, __nv_bfloat16* out,
            int N, int K, int R, int iters, int nsets) {
  constexpr int RT = 2, KS = 4, U = 4;
  const int GPR = K / 64, J = 1;
  size_t sm = J * K * 2 + J * GPR * 4 + KS * RT * 16 * 4 + J * 8;
  long w_sn = K / 8, w_se = (long)N * w_sn, s_sn = GPR, s_se = (long)N * GPR;
  cudaStream_t st; cudaStreamCreate(&st);
  cudaGraph_t g; cudaGraphExec_t ge;
  cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < iters; ++i)
    qmv4_mma_kernel<RT, KS, U, MODE><<<dim3(N / (RT * 16), R), 256, sm, st>>>(x, w, s, b, ids + (i % nsets) * R, nullptr, out, N, K, J, R,
                                                                         w_se, w_sn, s_se, s_sn, s_se, s_sn);
  cudaStreamEndCapture(st, &g); cudaGraphInstantiate(&ge, g, 0);
  cudaGraphLaunch(ge, st); cudaStreamSynchronize(st);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, st); cudaGraphLaunch(ge, st); cudaEventRecord(e1, st); cudaStreamSynchronize(st);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  return ms * 1e3f / iters;
}
int main() {
  const int E = 512, N = 1280, K = 2560, iters = 400;
  size_t wbytes = (size_t)E * N * K / 2;
  void *w, *s, *b, *x, *out; int* ids;
  cudaMalloc(&w, wbytes); cudaMemset(w, 0x5a, wbytes);
  cudaMalloc(&s, (size_t)E * N * (K / 64) * 2); cudaMemset(s, 0x3c, (size_t)E * N * (K / 64) * 2);
  cudaMalloc(&b, (size_t)E * N * (K / 64) * 2); cudaMemset(b, 0x3c, (size_t)E * N * (K / 64) * 2);
  cudaMalloc(&x, 64 * K * 2); cudaMemset(x, 0x3c, 64 * K * 2);
  cudaMalloc(&out, 64 * N * 2);
  for (int R : {10, 30}) {
    std::vector<int> h(iters * R);
    for (int i = 0; i < iters; ++i) for (int r = 0; r < R; ++r) h[i * R + r] = (i * 37 + r * 51 + (i * r) % 7) % E;
    cudaMalloc(&ids, h.size() * 4); cudaMemcpy(ids, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    auto run = [&](auto f, const char* name) {
      for (int nsets : {iters, 1}) printf("R=%d %-28s %s %7.2f us\n", R, name, nsets == 1 ? "L2  " : "HBM ", f(nsets));
    };
#define B(M, NAME) run([&](int ns) { return bench<M>((__nv_bfloat16*)x, (uint2*)w, (__nv_bfloat16*)s, (__nv_bfloat16*)b, ids, (__nv_bfloat16*)out, N, K, R, iters, ns); }, NAME)
    B(0, "full");
    B(1, "no scale/bias loads");
    B(2, "no weight loads");
    B(3, "no weight, no scale loads");
    B(7, "+ no prologue");
    B(15, "+ no mma");
    B(4, "full minus prologue");
    B(8, "full minus mma");
    cudaFree(ids);
  }
  cudaError_t err = cudaGetLastError(); printf("%s\n", cudaGetErrorString(err));
}
