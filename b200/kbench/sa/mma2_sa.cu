#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include "mma2.cuh"
template <int RT, int KS, int U>
float bench(const __nv_bfloat16* x, const uint2* w, const __nv_bfloat16* s, const __nv_bfloat16* b, int* ids, const float* wts,
            __nv_bfloat16* out, int E, int N, int K, int O, int J, int x_div, int iters, int nsets) {
  long w_sn = K / 8, w_se = (long)N * w_sn, s_sn = K / 64, s_se = (long)N * s_sn;
  cudaStream_t st; cudaStreamCreate(&st);
  cudaGraph_t g; cudaGraphExec_t ge;
  cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < iters; ++i)
    qmv4_mma2_kernel<RT, KS, U><<<dim3(N / (RT * 16), O), 256, 0, st>>>(x, w, s, b, ids + (i % nsets) * O * J, wts, out, N, K, J, x_div,
                                                                    w_se, w_sn, s_se, s_sn, s_se, s_sn);
  cudaStreamEndCapture(st, &g); cudaGraphInstantiate(&ge, g, 0);
  cudaGraphLaunch(ge, st); cudaStreamSynchronize(st);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, st); cudaGraphLaunch(ge, st); cudaEventRecord(e1, st); cudaStreamSynchronize(st);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(st);
  return ms * 1e3f / iters;
}
int main() {
  const int E = 512, iters = 400;
  void *w, *s, *b, *x, *out, *wts; int* ids;
  size_t wb = (size_t)E * 1280 * 2560 / 2, sb = (size_t)E * 1280 * 40 * 2;
  cudaMalloc(&w, wb); cudaMemset(w, 0x5a, wb); cudaMalloc(&s, sb); cudaMemset(s, 0x3c, sb); cudaMalloc(&b, sb); cudaMemset(b, 0x3c, sb);
  cudaMalloc(&x, 64 * 2560 * 2); cudaMemset(x, 0x3c, 64 * 2560 * 2); cudaMalloc(&out, 64 * 2560 * 2);
  cudaMalloc(&wts, 64 * 4); cudaMemset(wts, 0, 64 * 4);
  for (int T : {1, 3}) {
    const int R = 10 * T;
    std::vector<int> h(iters * R);
    for (int i = 0; i < iters; ++i) for (int r = 0; r < R; ++r) h[i * R + r] = (i * 37 + r * 51 + (i * r) % 7) % E;
    cudaMalloc(&ids, h.size() * 4); cudaMemcpy(ids, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    auto P = [&](const char* nm, float us, double mb) { printf("T=%d %-22s %7.2f us %6.0f GB/s\n", T, nm, us, mb / us * 1e-3); };
    const double gu = R * 1280.0 * (1280 + 160), dn = R * 2560.0 * (320 + 40);
    // gate_up: N=1280 K=2560, O=R J=1 x_div=10
#define GU(RT, KS, U) P("gu RT" #RT " KS" #KS " U" #U, bench<RT, KS, U>((__nv_bfloat16*)x, (uint2*)w, (__nv_bfloat16*)s, (__nv_bfloat16*)b, ids, nullptr, (__nv_bfloat16*)out, E, 1280, 2560, R, 1, 10, iters, iters), gu)
#define DN(RT, KS, U) P("dn RT" #RT " KS" #KS " U" #U, bench<RT, KS, U>((__nv_bfloat16*)x, (uint2*)w, (__nv_bfloat16*)s, (__nv_bfloat16*)b, ids, (float*)wts, (__nv_bfloat16*)out, E, 2560, 640, T, 10, 1, iters, iters), dn)
    GU(1, 8, 5); GU(2, 4, 10); GU(1, 8, 8); GU(2, 4, 5); GU(4, 2, 5); GU(8, 1, 8);
    DN(1, 8, 13); DN(1, 8, 7); DN(1, 8, 4); DN(2, 4, 13); DN(2, 4, 7);
    cudaFree(ids);
  }
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
