// v4 tensor-core expert matvec: one round of loads per chunk of U items, no shared-memory staging of the activation.
// Same contract as qmv4_kernel: out[o, n] = sum_{j<J} rw(r) * dot(x[r / x_div], dequant(w[ids[r], n])), r = o * J + j.
// Block = 8 warps = RT row tiles (16 rows) x KS splits of the (route, 64-code group) item space.
// Lane q = lane % 4 owns 16 codes (8 bytes) of rows lane/4 and lane/4 + 8 per item; lanes 0..3 also hold the
// activation (B column 0) for those codes, read straight from global as two uint4 and permuted to the (0,4) (1,5)
// (2,6) (3,7) pair order the codes dequantize to, and their quad-reduced sum gives the group sum.
#pragma once
__device__ __forceinline__ uint32_t code_pair4(uint32_t w, int k) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(w >> (4 * k)), "r"(0x000F000Fu), "r"(0x43004300u));
  return r;
}
__device__ __forceinline__ void mma_bf16_4(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float bsum8(uint4 v) {  // sum of 8 bf16
  float t = 0.f;
  const uint32_t a[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) t += __uint_as_float(a[i] << 16) + __uint_as_float(a[i] & 0xFFFF0000u);
  return t;
}

template <int RT, int KS, int U>
__global__ void __launch_bounds__(256) qmv4_mma2_kernel(
    const __nv_bfloat16* __restrict__ x, const uint2* __restrict__ w, const __nv_bfloat16* __restrict__ s,
    const __nv_bfloat16* __restrict__ b, const int* __restrict__ ids, const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ out, int N, int K, int J, int x_div, long w_se, long w_sn, long s_se, long s_sn,
    long b_se, long b_sn) {
  __shared__ float red[KS > 1 ? KS * RT * 16 : 1];
  const int GPR = K / 64, o = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int rt = warp % RT, ks = warp / RT;
  const int n0 = (blockIdx.x * RT + rt) * 16;
  const int q = lane & 3, rlo = n0 + (lane >> 2), rhi = rlo + 8;
  const bool col0 = lane < 4;
  const int items = J * GPR;
  float acc_lo = 0.f, acc_hi = 0.f;
  for (int base = ks; base < items; base += KS * U) {
    uint2 wlo[U], whi[U];
    uint4 xa[U], xb[U];
    float slo[U], shi[U], blo[U], bhi[U], rw[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int it = base + KS * u;
      if (it < items) {
        const int j = it / GPR, g = it - j * GPR, r = o * J + j;
        const long e = __ldg(ids + r);
        rw[u] = weights ? __ldg(weights + r) : 1.f;
        const long off = e * w_se / 2 + g * 4 + q;
        wlo[u] = __ldg(w + off + (long)rlo * w_sn / 2);
        whi[u] = __ldg(w + off + (long)rhi * w_sn / 2);
        if (col0) {
          const uint4* xp = reinterpret_cast<const uint4*>(x + (long)(r / x_div) * K + g * 64 + q * 16);
          xa[u] = __ldg(xp); xb[u] = __ldg(xp + 1);
        } else {
          xa[u] = xb[u] = make_uint4(0, 0, 0, 0);
        }
        if (q == 0) {
          slo[u] = __bfloat162float(__ldg(s + e * s_se + rlo * s_sn + g)); blo[u] = __bfloat162float(__ldg(b + e * b_se + rlo * b_sn + g));
          shi[u] = __bfloat162float(__ldg(s + e * s_se + rhi * s_sn + g)); bhi[u] = __bfloat162float(__ldg(b + e * b_se + rhi * b_sn + g));
        }
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int it = base + KS * u;
      if (it >= items) break;
      float d[4] = {0.f, 0.f, 0.f, 0.f};
      const uint32_t lo[2] = {wlo[u].x, wlo[u].y}, hi[2] = {whi[u].x, whi[u].y};
      const uint4 xv[2] = {xa[u], xb[u]};
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t p0 = __byte_perm(xv[h].x, xv[h].z, 0x5410), p1 = __byte_perm(xv[h].x, xv[h].z, 0x7632);
        const uint32_t p2 = __byte_perm(xv[h].y, xv[h].w, 0x5410), p3 = __byte_perm(xv[h].y, xv[h].w, 0x7632);
        mma_bf16_4(d, code_pair4(lo[h], 0), code_pair4(hi[h], 0), code_pair4(lo[h], 1), code_pair4(hi[h], 1), p0, p1);
        mma_bf16_4(d, code_pair4(lo[h], 2), code_pair4(hi[h], 2), code_pair4(lo[h], 3), code_pair4(hi[h], 3), p2, p3);
      }
      float sg = bsum8(xv[0]) + bsum8(xv[1]);  // lanes 0..3: their 16 codes' activation sum; lanes >= 4 hold zeros
      sg += __shfl_xor_sync(0xffffffffu, sg, 1);
      sg += __shfl_xor_sync(0xffffffffu, sg, 2);
      sg = __shfl_sync(0xffffffffu, sg, 0);
      if (q == 0) {
        acc_lo = fmaf(rw[u], fmaf(slo[u], d[0] - 128.f * sg, blo[u] * sg), acc_lo);
        acc_hi = fmaf(rw[u], fmaf(shi[u], d[2] - 128.f * sg, bhi[u] * sg), acc_hi);
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
