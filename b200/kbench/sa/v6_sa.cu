#include <cstdio>
#include <vector>
#include <cmath>
#include "v6.cuh"
template <int K, int JM, int RT, int ST>
float bench(const __nv_bfloat16* x, const uint4* w, const __nv_bfloat16* s, const __nv_bfloat16* b, int* ids, const float* wts,
            __nv_bfloat16* out, int N, int O, int J, int x_div, int iters, bool once = false) {
  constexpr size_t sm = v6::smem_bytes<K, JM, RT>();
  auto kf = v6::kernel<K, JM, RT, ST>;
  cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  if (once) { kf<<<dim3(N / (16 * RT), O), 256, sm>>>(x, w, s, b, ids, wts, out, N, J, x_div); cudaDeviceSynchronize(); return 0; }
  cudaStream_t st; cudaStreamCreate(&st); cudaGraph_t g; cudaGraphExec_t ge;
  cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < iters; ++i) kf<<<dim3(N / (16 * RT), O), 256, sm, st>>>(x, w, s, b, ids + i * O * J, wts, out, N, J, x_div);
  cudaStreamEndCapture(st, &g); cudaGraphInstantiate(&ge, g, 0);
  cudaGraphLaunch(ge, st); cudaStreamSynchronize(st);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, st); cudaGraphLaunch(ge, st); cudaEventRecord(e1, st); cudaStreamSynchronize(st);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(st);
  return ms * 1e3f / iters;
}
// CPU reference for one output row block
float bf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
int main() {
  const int E = 512, iters = 400;
  size_t wb = (size_t)E * 1280 * 2560 / 2, sb = (size_t)E * 1280 * 40 * 2;
  std::vector<uint32_t> hw(wb / 4); uint32_t z = 1; for (auto& v : hw) { z ^= z << 13; z ^= z >> 17; z ^= z << 5; v = z; }
  std::vector<uint16_t> hs(sb / 2), hb(sb / 2), hx(64 * 2560);
  for (auto& v : hs) { z ^= z << 13; z ^= z >> 17; z ^= z << 5; v = 0x3c00 + (z & 0x3ff) - 0x200; }
  for (auto& v : hb) { z ^= z << 13; z ^= z >> 17; z ^= z << 5; v = 0xbc00 + (z & 0x3ff) - 0x200; }
  for (auto& v : hx) { z ^= z << 13; z ^= z >> 17; z ^= z << 5; v = ((z & 1) ? 0xbf00 : 0x3f00) + (z >> 8 & 0xff) - 0x80; }
  void *w, *s, *b, *x, *out, *wts; int* ids;
  cudaMalloc(&w, wb); cudaMemcpy(w, hw.data(), wb, cudaMemcpyHostToDevice);
  cudaMalloc(&s, sb); cudaMemcpy(s, hs.data(), sb, cudaMemcpyHostToDevice);
  cudaMalloc(&b, sb); cudaMemcpy(b, hb.data(), sb, cudaMemcpyHostToDevice);
  cudaMalloc(&x, hx.size() * 2); cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
  cudaMalloc(&out, 64 * 2560 * 2);
  std::vector<float> hwt(64); for (int i = 0; i < 64; ++i) hwt[i] = 0.05f * (i % 10 + 1);
  cudaMalloc(&wts, 64 * 4); cudaMemcpy(wts, hwt.data(), 64 * 4, cudaMemcpyHostToDevice);
  for (int T : {1, 3}) {
    const int R = 10 * T;
    std::vector<int> h(iters * R);
    for (int i = 0; i < iters; ++i) for (int r = 0; r < R; ++r) h[i * R + r] = (i * 37 + r * 51 + (i * r) % 7) % E;
    cudaMalloc(&ids, h.size() * 4); cudaMemcpy(ids, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    // correctness: gate_up (N=1280, K=2560) then down (N=2560, K=640, banks reinterpreted)
    auto check = [&](int N, int K, int O, int J, int xdiv, bool wsum) {
      std::vector<uint16_t> ho(O * N); cudaMemcpy(ho.data(), out, O * N * 2, cudaMemcpyDeviceToHost);
      double num = 0, den = 0;
      for (int o = 0; o < O; ++o) for (int n = 0; n < N; n += 37) {
        double acc = 0;
        for (int j = 0; j < J; ++j) {
          int r = o * J + j, e = h[r];
          double dot = 0;
          for (int k = 0; k < K; ++k) {
            uint32_t word = hw[((size_t)e * N + n) * (K / 8) + k / 8];
            int c = (word >> (4 * (k % 8))) & 15;
            dot += (c * bf(hs[((size_t)e * N + n) * (K / 64) + k / 64]) + bf(hb[((size_t)e * N + n) * (K / 64) + k / 64])) * bf(hx[(size_t)(r / xdiv) * K + k]);
          }
          acc += (wsum ? hwt[r] : 1.0) * dot;
        }
        double got = bf(ho[o * N + n]); num += (got - acc) * (got - acc); den += acc * acc;
      }
      return sqrt(num / den);
    };
    const double gu = R * 1280.0 * (1280 + 160), dn = R * 2560.0 * (320 + 40);
    bench<2560, 1, 1, 4>((__nv_bfloat16*)x, (uint4*)w, (__nv_bfloat16*)s, (__nv_bfloat16*)b, ids, nullptr, (__nv_bfloat16*)out, 1280, R, 1, 10, 1, true);
    printf("T=%d gate_up rel err %.2e\n", T, check(1280, 2560, R, 1, 10, false));
    bench<640, 10, 1, 2>((__nv_bfloat16*)x, (uint4*)w, (__nv_bfloat16*)s, (__nv_bfloat16*)b, ids, (float*)wts, (__nv_bfloat16*)out, 2560, T, 10, 1, 1, true);
    printf("T=%d down    rel err %.2e\n", T, check(2560, 640, T, 10, 1, true));
#define GU(RT, ST) { float us = bench<2560, 1, RT, ST>((__nv_bfloat16*)x, (uint4*)w, (__nv_bfloat16*)s, (__nv_bfloat16*)b, ids, nullptr, (__nv_bfloat16*)out, 1280, R, 1, 10, iters); printf("T=%d gu RT%d ST%d %6.2f us %5.0f GB/s\n", T, RT, ST, us, gu / us * 1e-3); }
#define DN(RT, ST) { float us = bench<640, 10, RT, ST>((__nv_bfloat16*)x, (uint4*)w, (__nv_bfloat16*)s, (__nv_bfloat16*)b, ids, (float*)wts, (__nv_bfloat16*)out, 2560, T, 10, 1, iters); printf("T=%d dn RT%d ST%d %6.2f us %5.0f GB/s\n", T, RT, ST, us, dn / us * 1e-3); }
    GU(1, 1); GU(2, 2); { float us = bench<2560, 1, 1, 1>((__nv_bfloat16*)x, (uint4*)w, (__nv_bfloat16*)s, (__nv_bfloat16*)b, ids, nullptr, (__nv_bfloat16*)out, 1280, R, 1, 1, iters); printf("T=%d gu distinct-x %6.2f us\n", T, us); }
    DN(1, 1);
    cudaFree(ids);
  }
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
