// Load-mechanism comparison for R=10 random experts' 16-row gate_up tiles (16 x 1280 B = 20 KB contiguous per block):
// A: __ldg to registers; B: cp.async.cg 16 B to smem; C: cp.async.bulk (TMA 1D) per row into padded smem rows; D: one bulk copy.
#include <cstdio>
#include <cstdint>
#include <vector>
constexpr int ROWB = 1280, ROWS = 16;
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(smem)), "l"(gmem));
}
template <int MODE>
__global__ void __launch_bounds__(256) k(const unsigned char* __restrict__ w, const int* __restrict__ ids, long eb, unsigned* out) {
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) unsigned long long bar;
  const int r = blockIdx.y;
  const long e = __ldg(ids + r);
  const unsigned char* p = w + e * eb + (long)blockIdx.x * ROWS * ROWB;
  uint32_t acc = 0;
  if (MODE == 0) {
    const uint4* q = reinterpret_cast<const uint4*>(p);
    uint4 v[5];
#pragma unroll
    for (int u = 0; u < 5; ++u) v[u] = __ldg(q + threadIdx.x + 256 * u);
#pragma unroll
    for (int u = 0; u < 5; ++u) acc ^= v[u].x ^ v[u].w;
  } else if (MODE == 1) {
#pragma unroll
    for (int u = 0; u < 5; ++u) { int i = threadIdx.x + 256 * u; int row = i / 80, c = i % 80; cp_async16(sm + row * (ROWB + 16) + c * 16, p + i * 16); }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    acc = sm[threadIdx.x * 7];
  } else {
    const unsigned ba = (unsigned)__cvta_generic_to_shared(&bar);
    if (threadIdx.x == 0) {
      asm volatile("mbarrier.init.shared.b64 [%0], 1;" ::"r"(ba));
      asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(ba), "r"(ROWS * ROWB));
      if (MODE == 2) {
        for (int row = 0; row < ROWS; ++row)
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                       ::"r"((unsigned)__cvta_generic_to_shared(sm + row * (ROWB + 16))), "l"(p + row * ROWB), "r"(ROWB), "r"(ba) : "memory");
      } else {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     ::"r"((unsigned)__cvta_generic_to_shared(sm)), "l"(p), "r"(ROWS * ROWB), "r"(ba) : "memory");
      }
    }
    asm volatile("{\n.reg .pred P;\nW: mbarrier.try_wait.parity.shared.b64 P, [%0], 0;\n@!P bra W;\n}" ::"r"(ba) : "memory");
    acc = sm[threadIdx.x * 7];
  }
  if (acc == 0x12345678u) out[0] = acc;
}
template <int M>
float bench(const unsigned char* w, int* ids, long eb, int R, int iters, unsigned* out) {
  auto kf = k<M>; cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * 1024);
  cudaStream_t st; cudaStreamCreate(&st); cudaGraph_t g; cudaGraphExec_t ge;
  cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < iters; ++i) kf<<<dim3(eb / (ROWS * ROWB), R), 256, 32 * 1024, st>>>(w, ids + i * R, eb, out);
  cudaStreamEndCapture(st, &g); cudaGraphInstantiate(&ge, g, 0);
  cudaGraphLaunch(ge, st); cudaStreamSynchronize(st);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, st); cudaGraphLaunch(ge, st); cudaEventRecord(e1, st); cudaStreamSynchronize(st);
  float ms; cudaEventElapsedTime(&ms, e0, e1); return ms * 1e3f / iters;
}
int main() {
  const int E = 512, iters = 400; const long eb = 1280L * ROWB;
  unsigned char* w; cudaMalloc(&w, E * eb); cudaMemset(w, 1, E * eb);
  unsigned* out; cudaMalloc(&out, 4);
  for (int R : {10, 30}) {
    std::vector<int> h(iters * R);
    for (int i = 0; i < iters; ++i) for (int r = 0; r < R; ++r) h[i * R + r] = (i * 37 + r * 51 + (i * r) % 7) % E;
    int* ids; cudaMalloc(&ids, h.size() * 4); cudaMemcpy(ids, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    printf("R=%d ldg %.2f | cp.async %.2f | bulk/row %.2f | bulk tile %.2f us\n", R, bench<0>(w, ids, eb, R, iters, out),
           bench<1>(w, ids, eb, R, iters, out), bench<2>(w, ids, eb, R, iters, out), bench<3>(w, ids, eb, R, iters, out));
  }
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
