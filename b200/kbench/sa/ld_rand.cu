// Pure-load bound: read R random experts (each a contiguous block of BYTES) and xor-reduce, one block per CHUNK bytes.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstdlib>
template <int U>
__global__ void __launch_bounds__(256) rd(const uint4* __restrict__ w, const int* __restrict__ ids, long expert_u4, int chunk_u4, unsigned* out) {
  const int r = blockIdx.y;
  const long e = __ldg(ids + r);
  const uint4* p = w + e * expert_u4 + (long)blockIdx.x * chunk_u4;
  uint32_t acc = 0;
  for (int i = threadIdx.x; i < chunk_u4; i += 256 * U) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = (i + 256 * u < chunk_u4) ? __ldg(p + i + 256 * u) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x12345678u) out[0] = acc;
}
template <int U>
float bench(const uint4* w, int* ids, long expert_u4, int R, int chunk_u4, int iters, unsigned* out) {
  cudaStream_t st; cudaStreamCreate(&st); cudaGraph_t g; cudaGraphExec_t ge;
  cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < iters; ++i) rd<U><<<dim3(expert_u4 / chunk_u4, R), 256, 0, st>>>(w, ids + i * R, expert_u4, chunk_u4, out);
  cudaStreamEndCapture(st, &g); cudaGraphInstantiate(&ge, g, 0);
  cudaGraphLaunch(ge, st); cudaStreamSynchronize(st);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, st); cudaGraphLaunch(ge, st); cudaEventRecord(e1, st); cudaStreamSynchronize(st);
  float ms; cudaEventElapsedTime(&ms, e0, e1); return ms * 1e3f / iters;
}
int main() {
  const int E = 512, iters = 400; const long expert_u4 = 1280L * 1280 / 16;  // gate_up codes 1.6 MB
  uint4* w; cudaMalloc(&w, E * expert_u4 * 16); cudaMemset(w, 1, E * expert_u4 * 16); if (getenv("RAND")) { std::vector<uint32_t> hr(E * expert_u4 * 4); uint32_t z = 12345; for (auto& v : hr) { z ^= z << 13; z ^= z >> 17; z ^= z << 5; v = z; } cudaMemcpy(w, hr.data(), hr.size() * 4, cudaMemcpyHostToDevice); }
  unsigned* out; cudaMalloc(&out, 4);
  for (int R : {10, 30}) {
    std::vector<int> h(iters * R);
    for (int i = 0; i < iters; ++i) for (int r = 0; r < R; ++r) h[i * R + r] = (i * 37 + r * 51 + (i * r) % 7) % E;
    int* ids; cudaMalloc(&ids, h.size() * 4); cudaMemcpy(ids, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    for (int chunk : {1024, 2048, 4096, 8192, 20480}) {
      double mb = R * expert_u4 * 16.0;
      float a = bench<1>(w, ids, expert_u4, R, chunk, iters, out), b = bench<4>(w, ids, expert_u4, R, chunk, iters, out), c = bench<8>(w, ids, expert_u4, R, chunk, iters, out);
      printf("R=%d chunk %6d B (%4ld blocks): U1 %6.2f us %5.0f GB/s | U4 %6.2f us %5.0f GB/s | U8 %6.2f us %5.0f GB/s\n", R, chunk * 16,
             R * expert_u4 / chunk, a, mb / a * 1e-3, b, mb / b * 1e-3, c, mb / c * 1e-3);
    }
  }
}
