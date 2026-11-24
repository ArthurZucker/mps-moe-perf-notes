#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
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

template <int JMAX, int MODE>
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
    if (!(MODE & 2)) cp_async16(ws_ + (j * 16 + r) * RB + c * 16, w + (long)e * (w_se / 4) + (long)(n0 + r) * CPR + c);
  }
  const int SCH = 16 * GPR * 2 / 16;  // 16-byte chunks of one route's 16-row scale tile
  for (int i = threadIdx.x; i < 2 * J * SCH; i += 256) {
    const int which = i / (J * SCH), rem = i - which * J * SCH, j = rem / SCH, c = rem - j * SCH;
    int e = e_[0];
#pragma unroll
    for (int jj = 1; jj < JMAX; ++jj) if (jj == j) e = e_[jj];
    const __nv_bfloat16* src = (which ? b + e * b_se : s + e * s_se) + (long)n0 * GPR;
    if (!(MODE & 8)) cp_async16((which ? bs : ss) + j * 16 * GPR + c * 8, src + c * 8);
  }
  for (int i = threadIdx.x; i < J * K / 8; i += 256) {
    const int j = i / (K / 8), c = i - j * (K / 8);
    if (!(MODE & 4)) cp_async16(xs + i, reinterpret_cast<const uint4*>(x + (long)((o * J + j) / x_div) * K) + c);
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  if (MODE & 1) { if (threadIdx.x == 0 && ws_[5] == 77 && ws_[9] == 13) out[0] = __float2bfloat16(1.f); return; }
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


template <int JM, int MODE>
float bench(const __nv_bfloat16* x, const uint4* w, const __nv_bfloat16* s, const __nv_bfloat16* b, int* ids, const float* wts,
            __nv_bfloat16* out, int N, int K, int O, int J, int x_div, int iters) {
  const int GPR = K / 64;
  size_t sm = J * 16 * (K / 2 + 16) + 2 * J * 16 * GPR * 2 + 16 + 8 * 16 * 4 + J * K * 2;
  auto kf = qmv4_tc_kernel<JM, MODE>;
  cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  cudaStream_t st; cudaStreamCreate(&st); cudaGraph_t g; cudaGraphExec_t ge;
  cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < iters; ++i)
    kf<<<dim3(N / 16, O), 256, sm, st>>>(x, w, s, b, ids + i * O * J, wts, out, N, K, J, x_div, (long)N * K / 8, (long)N * GPR, (long)N * GPR);
  cudaStreamEndCapture(st, &g); cudaGraphInstantiate(&ge, g, 0);
  cudaGraphLaunch(ge, st); cudaStreamSynchronize(st);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, st); cudaGraphLaunch(ge, st); cudaEventRecord(e1, st); cudaStreamSynchronize(st);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  return ms * 1e3f / iters;
}
int main() {
  const int E = 512, iters = 400;
  void *w, *s, *b, *x, *out, *wts; int* ids;
  size_t wb = (size_t)E * 1280 * 2560 / 2, sb = (size_t)E * 1280 * 40 * 2;
  cudaMalloc(&w, wb); cudaMemset(w, 0x5a, wb); cudaMalloc(&s, sb); cudaMemset(s, 0x3c, sb); cudaMalloc(&b, sb); cudaMemset(b, 0x3c, sb);
  cudaMalloc(&x, 64 * 2560 * 2); cudaMemset(x, 0x3c, 64 * 2560 * 2); cudaMalloc(&out, 64 * 2560 * 2);
  cudaMalloc(&wts, 64 * 4); cudaMemset(wts, 0, 64 * 4);
  const int R = 10;
  std::vector<int> h(iters * R);
  for (int i = 0; i < iters; ++i) for (int r = 0; r < R; ++r) h[i * R + r] = (i * 37 + r * 51 + (i * r) % 7) % E;
  cudaMalloc(&ids, h.size() * 4); cudaMemcpy(ids, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
#define GU(M) printf("gu MODE %d %7.2f us\n", M, bench<1, M>((__nv_bfloat16*)x, (uint4*)w, (__nv_bfloat16*)s, (__nv_bfloat16*)b, ids, nullptr, (__nv_bfloat16*)out, 1280, 2560, R, 1, 10, iters))
#define DN(M) printf("dn MODE %d %7.2f us\n", M, bench<10, M>((__nv_bfloat16*)x, (uint4*)w, (__nv_bfloat16*)s, (__nv_bfloat16*)b, ids, (float*)wts, (__nv_bfloat16*)out, 2560, 640, 1, 10, 1, iters))
  GU(0); GU(1); GU(5); GU(9); GU(13); GU(15); GU(3); DN(0); DN(1); DN(5); DN(9); DN(13); DN(15);
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
