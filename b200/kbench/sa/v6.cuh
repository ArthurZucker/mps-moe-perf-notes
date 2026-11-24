// v6 tensor-core expert matvec (group size 64, resident banks). Contract as qmv4_kernel:
//   out[o, n] = sum_{j<J} rw(r) * dot(x[r / x_div], dequant(w[ids[r], n])),  r = o * J + j.
// Block: 16 * RT rows of output o, 8 warps = RT row tiles x KS = 8 / RT splits of the (route, group) items.
// Loads: one cp.async batch after the ids -- activations + scales / biases (commit group 0), then the weight rows in
// STAGES commit groups by group range, so computing stage s overlaps the copies of the later stages.
// Prep (once per block): activations permuted in place to the (0,4)(1,5)(2,6)(3,7) bf16-pair order and group sums.
// Per item: 2 LDS.64 codes, 2 LDS.128 activations (lanes 0..3 = B column 0), 16 lop3 dequants (bf16 128 + c),
// 4 mma.m16n8k16; dot = scale * (D - 128 * gsum) + bias * gsum.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>
namespace v6 {
__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ uint32_t pair(uint32_t w, int k) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(w >> (4 * k)), "r"(0x000F000Fu), "r"(0x43004300u));
  return r;
}
__device__ __forceinline__ void mma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <int STAGES> __device__ __forceinline__ void wait_stage(int s) {
  // commit groups: 0 = activations + scales, 1 + s = weight stage s; wait until group 1 + s landed
  switch (STAGES - 1 - s) {
    case 0: wait_group<0>(); break;
    case 1: wait_group<1>(); break;
    case 2: wait_group<2>(); break;
    case 3: wait_group<3>(); break;
    default: wait_group<4>(); break;
  }
}

template <int K, int JMAX, int RT, int STAGES>
__global__ void __launch_bounds__(256) kernel(
    const __nv_bfloat16* __restrict__ x, const uint4* __restrict__ w, const __nv_bfloat16* __restrict__ s,
    const __nv_bfloat16* __restrict__ b, const int* __restrict__ ids, const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ out, int N, int J, int x_div) {
  constexpr int GPR = K / 64, CPR = K / 32, RB = K / 2 + 16, ROWS = 16 * RT, KS = 8 / RT, GPS = GPR / STAGES;
  static_assert(GPR % STAGES == 0, "stages divide the groups");
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* wsm = smem;                                                     // [J][ROWS][RB]
  __nv_bfloat16* ssm = reinterpret_cast<__nv_bfloat16*>(smem + JMAX * ROWS * RB); // [J][ROWS][GPR]
  __nv_bfloat16* bsm = ssm + JMAX * ROWS * GPR;
  uint4* xsm = reinterpret_cast<uint4*>(bsm + JMAX * ROWS * GPR);               // [J][K / 8]
  float* gsm = reinterpret_cast<float*>(xsm + JMAX * K / 8);                     // [J][GPR]
  float* red = gsm + JMAX * GPR;                                                 // [8][16]
  const int o = blockIdx.y, n0 = blockIdx.x * ROWS, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int e_[JMAX];
#pragma unroll
  for (int j = 0; j < JMAX; ++j) e_[j] = j < J ? __ldg(ids + o * J + j) : 0;
  // group 0: activations, scales, biases
#pragma unroll
  for (int j = 0; j < JMAX; ++j) {
    if (j >= J) break;
    const uint4* xr = reinterpret_cast<const uint4*>(x + (long)((o * J + j) / x_div) * K);
    for (int c = tid; c < K / 8; c += 256) cp16(xsm + j * (K / 8) + c, xr + c);
    constexpr int SC = ROWS * GPR / 8;  // 16-byte chunks of a route's scale tile
    const __nv_bfloat16* sp = s + (long)e_[j] * N * GPR + (long)n0 * GPR;
    const __nv_bfloat16* bp = b + (long)e_[j] * N * GPR + (long)n0 * GPR;
    for (int c = tid; c < 2 * SC; c += 256)
      cp16(c < SC ? ssm + j * ROWS * GPR + c * 8 : bsm + j * ROWS * GPR + (c - SC) * 8, c < SC ? sp + c * 8 : bp + (c - SC) * 8);
  }
  commit();
  // groups 1..STAGES: weight rows, stage st = groups [st * GPS, (st + 1) * GPS) = 16-byte chunks [st * 2 * GPS, ...)
#pragma unroll
  for (int st = 0; st < STAGES; ++st) {
#pragma unroll
    for (int j = 0; j < JMAX; ++j) {
      if (j >= J) break;
      const uint4* wp = w + (long)e_[j] * N * CPR + (long)n0 * CPR;
      constexpr int CH = 2 * GPS;  // chunks per row per stage
      for (int i = tid; i < ROWS * CH; i += 256) {
        const int r = i / CH, c = st * CH + i % CH;
        cp16(wsm + (j * ROWS + r) * RB + c * 16, wp + r * CPR + c);
      }
    }
    commit();
  }
  wait_group<STAGES>();
  __syncthreads();
  // prep: permute activations to pair order, group sums
  for (int i = tid; i < J * K / 8; i += 256) {
    uint4 v = xsm[i], p;
    p.x = __byte_perm(v.x, v.z, 0x5410); p.y = __byte_perm(v.x, v.z, 0x7632);
    p.z = __byte_perm(v.y, v.w, 0x5410); p.w = __byte_perm(v.y, v.w, 0x7632);
    xsm[i] = p;
    float t = 0.f;
    const uint32_t a[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) t += __uint_as_float(a[k] << 16) + __uint_as_float(a[k] & 0xFFFF0000u);
    t += __shfl_xor_sync(0xffffffffu, t, 1);
    t += __shfl_xor_sync(0xffffffffu, t, 2);
    t += __shfl_xor_sync(0xffffffffu, t, 4);  // 8 chunks of 8 = one group of 64 (K / 8 is a multiple of 8, 256 too)
    if ((tid & 7) == 0) gsm[i / 8] = t;
  }
  const int rt = warp % RT, ks = warp / RT, q = lane & 3, rl = lane >> 2;
  const bool col0 = lane < 4;
  const int row_lo = rt * 16 + rl;
  float acc_lo = 0.f, acc_hi = 0.f;
#pragma unroll 1
  for (int st = 0; st < STAGES; ++st) {
    wait_stage<STAGES>(st);
    __syncthreads();
    for (int it = ks; it < J * GPS; it += KS) {
      const int j = it / GPS, g = st * GPS + it % GPS;
      const unsigned char* wr = wsm + (j * ROWS + row_lo) * RB + g * 32 + q * 8;
      const uint2 lo = *reinterpret_cast<const uint2*>(wr), hi = *reinterpret_cast<const uint2*>(wr + 8 * RB);
      uint4 xa = make_uint4(0, 0, 0, 0), xb = xa;
      if (col0) { const uint4* xp = xsm + j * (K / 8) + g * 8 + q * 2; xa = xp[0]; xb = xp[1]; }
      float d[4] = {0.f, 0.f, 0.f, 0.f};
      mma(d, pair(lo.x, 0), pair(hi.x, 0), pair(lo.x, 1), pair(hi.x, 1), xa.x, xa.y);
      mma(d, pair(lo.x, 2), pair(hi.x, 2), pair(lo.x, 3), pair(hi.x, 3), xa.z, xa.w);
      mma(d, pair(lo.y, 0), pair(hi.y, 0), pair(lo.y, 1), pair(hi.y, 1), xb.x, xb.y);
      mma(d, pair(lo.y, 2), pair(hi.y, 2), pair(lo.y, 3), pair(hi.y, 3), xb.z, xb.w);
      if (q == 0) {
        const float sg = gsm[j * GPR + g], rw = weights ? __ldg(weights + o * J + j) : 1.f;
        const int so = (j * ROWS + row_lo) * GPR + g;
        acc_lo = fmaf(rw, fmaf(__bfloat162float(ssm[so]), d[0] - 128.f * sg, __bfloat162float(bsm[so]) * sg), acc_lo);
        acc_hi = fmaf(rw, fmaf(__bfloat162float(ssm[so + 8 * GPR]), d[2] - 128.f * sg, __bfloat162float(bsm[so + 8 * GPR]) * sg), acc_hi);
      }
    }
  }
  if (KS == 1) {
    if (q == 0) { out[(long)o * N + n0 + row_lo] = __float2bfloat16(acc_lo); out[(long)o * N + n0 + row_lo + 8] = __float2bfloat16(acc_hi); }
    return;
  }
  if (q == 0) { red[(ks * RT + rt) * 16 + rl] = acc_lo; red[(ks * RT + rt) * 16 + rl + 8] = acc_hi; }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int k = 0; k < KS; ++k) v += red[(k * RT + tid / 16) * 16 + tid % 16];
    out[(long)o * N + n0 + tid] = __float2bfloat16(v);
  }
}
template <int K, int JMAX, int RT>
constexpr size_t smem_bytes() {
  return JMAX * 16 * RT * (K / 2 + 16) + 2 * JMAX * 16 * RT * (K / 64) * 2 + JMAX * K * 2 + JMAX * (K / 64) * 4 + 8 * 16 * 4;
}
}  // namespace v6
