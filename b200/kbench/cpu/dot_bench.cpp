// Single-thread cycles for one 4-bit affine row dot (group 64) with hot data. A: v2 (vpermw LUT). B: magic bf16 128+c.
#include <immintrin.h>
#include <x86intrin.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
inline float bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
inline uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
struct Lut { __m512i v; Lut() { alignas(64) uint16_t t[32]; for (int i = 0; i < 32; ++i) t[i] = f2bf((float)(i & 15)); v = _mm512_load_si512(t); } };
__attribute__((noinline)) float dotA(const uint8_t* wb, const uint16_t* s, const uint16_t* b, const uint16_t* x, const float* gs, int K, __m512i lut) {
  const __m512i m_lo = _mm512_set1_epi32(0x0F), m_hi = _mm512_set1_epi32(0xF0);
  __m512 acc = _mm512_setzero_ps(); float bacc = 0.f;
  for (int g = 0; g < K / 64; ++g) {
    const uint8_t* gb = wb + g * 32;
    __m512i i0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)gb)), i1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(gb + 16)));
    i0 = _mm512_or_si512(_mm512_and_si512(i0, m_lo), _mm512_slli_epi32(_mm512_and_si512(i0, m_hi), 12));
    i1 = _mm512_or_si512(_mm512_and_si512(i1, m_lo), _mm512_slli_epi32(_mm512_and_si512(i1, m_hi), 12));
    __m512 d = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh)_mm512_permutexvar_epi16(i0, lut), (__m512bh)_mm512_loadu_si512(x + g * 64));
    d = _mm512_dpbf16_ps(d, (__m512bh)_mm512_permutexvar_epi16(i1, lut), (__m512bh)_mm512_loadu_si512(x + g * 64 + 32));
    acc = _mm512_fmadd_ps(d, _mm512_set1_ps(bf2f(s[g])), acc);
    bacc = fmaf(bf2f(b[g]), gs[g], bacc);
  }
  return _mm512_reduce_add_ps(acc) + bacc;
}
// B: xp = x permuted per 128-code chunk: xp[c][k][i] (bf16 pair) = (x[c*128 + 8i + 2k], x[c*128 + 8i + 2k + 1]), i < 16, k < 4
// lane i of the 64-byte chunk load = word i = codes 8i..8i+7 (group = i / 8)
template <int NG>
__attribute__((noinline)) float dotB(const uint8_t* wb, const uint16_t* s, const uint16_t* b, const uint32_t* xp, const float* gs) {
  const __m512i sel = _mm512_set1_epi32(0xFFFF0000), C = _mm512_set1_epi32(0x43004300), M = _mm512_set1_epi32(0x000F000F);
  // scales / corrected biases of the row as fp32 (NG groups)
  alignas(64) float sf[NG + 16], cf[NG + 16];
  float bsum = 0.f;
  for (int g = 0; g < NG; ++g) { sf[g] = bf2f(s[g]); bsum += (bf2f(b[g]) - 128.f * sf[g]) * gs[g]; }
  __m512 acc = _mm512_setzero_ps();
  const __m512i half = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
#pragma unroll 2
  for (int c = 0; c < NG / 2; ++c) {
    const __m512i w = _mm512_loadu_si512(wb + c * 64);
    __m512 d = _mm512_setzero_ps();
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const __m512i t = _mm512_srli_epi32(w, 8 * k);
      const __m512i a = _mm512_slli_epi32(t, 12);
      __m512i lane = _mm512_ternarylogic_epi32(sel, a, t, 0xCA);      // sel ? a : t
      lane = _mm512_ternarylogic_epi32(lane, M, C, 0xEA);              // (lane & M) | C
      d = _mm512_dpbf16_ps(d, (__m512bh)lane, (__m512bh)_mm512_loadu_si512(xp + c * 64 + k * 16));
    }
    const __m512 sv = _mm512_permutexvar_ps(half, _mm512_loadu_ps(sf + 2 * c));
    acc = _mm512_fmadd_ps(d, sv, acc);
  }
  return _mm512_reduce_add_ps(acc) + bsum;
}

// C: R rows at once (independent accumulator chains, shared activation loads)
template <int NG, int R>
__attribute__((noinline)) void dotC(const uint8_t* wb, size_t row_bytes, const uint16_t* s, const uint16_t* b, size_t sstride,
                                    const uint32_t* xp, const float* gs, float* out) {
  const __m512i sel = _mm512_set1_epi32(0xFFFF0000), C = _mm512_set1_epi32(0x43004300), M = _mm512_set1_epi32(0x000F000F);
  const __m512i half = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
  alignas(64) float sf[R][NG + 16];
  float bsum[R];
  for (int r = 0; r < R; ++r) {
    bsum[r] = 0.f;
    for (int g = 0; g < NG; ++g) { sf[r][g] = bf2f(s[r * sstride + g]); bsum[r] += (bf2f(b[r * sstride + g]) - 128.f * sf[r][g]) * gs[g]; }
  }
  __m512 acc[R];
  for (int r = 0; r < R; ++r) acc[r] = _mm512_setzero_ps();
  for (int c = 0; c < NG / 2; ++c) {
    __m512i w[R];
    __m512 d[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { w[r] = _mm512_loadu_si512(wb + r * row_bytes + c * 64); d[r] = _mm512_setzero_ps(); }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const __m512bh xv = (__m512bh)_mm512_loadu_si512(xp + c * 64 + k * 16);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const __m512i t = _mm512_srli_epi32(w[r], 8 * k);
        __m512i lane = _mm512_ternarylogic_epi32(sel, _mm512_slli_epi32(t, 12), t, 0xCA);
        lane = _mm512_ternarylogic_epi32(lane, M, C, 0xEA);
        d[r] = _mm512_dpbf16_ps(d[r], (__m512bh)lane, xv);
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = _mm512_fmadd_ps(d[r], _mm512_permutexvar_ps(half, _mm512_loadu_ps(sf[r] + 2 * c)), acc[r]);
  }
  for (int r = 0; r < R; ++r) out[r] = _mm512_reduce_add_ps(acc[r]) + bsum[r];
}

// D: pairs (nibble k, nibble k + 4) of each word: lane = ((w >> 4k) & 0x000F000F) | 0x43004300. xq[c][k][i] = (x[c*128 + 8i + k], x[c*128 + 8i + k + 4])
template <int NG, int R>
__attribute__((noinline)) void dotD(const uint8_t* wb, size_t row_bytes, const uint16_t* s, const uint16_t* b, size_t sstride,
                                    const uint32_t* xq, const float* gs, float* out) {
  const __m512i C = _mm512_set1_epi32(0x43004300), M = _mm512_set1_epi32(0x000F000F);
  const __m512i half = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
  alignas(64) float sf[R][NG + 16];
  float bsum[R];
  for (int r = 0; r < R; ++r) {
    bsum[r] = 0.f;
    for (int g = 0; g < NG; ++g) { sf[r][g] = bf2f(s[r * sstride + g]); bsum[r] += (bf2f(b[r * sstride + g]) - 128.f * sf[r][g]) * gs[g]; }
  }
  __m512 acc[R];
  for (int r = 0; r < R; ++r) acc[r] = _mm512_setzero_ps();
  for (int c = 0; c < NG / 2; ++c) {
    __m512i w[R];
    __m512 d[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { w[r] = _mm512_loadu_si512(wb + r * row_bytes + c * 64); d[r] = _mm512_setzero_ps(); }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const __m512bh xv = (__m512bh)_mm512_loadu_si512(xq + c * 64 + k * 16);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const __m512i t = k ? _mm512_srli_epi32(w[r], 4 * k) : w[r];
        d[r] = _mm512_dpbf16_ps(d[r], (__m512bh)_mm512_ternarylogic_epi32(t, M, C, 0xEA), xv);
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = _mm512_fmadd_ps(d[r], _mm512_permutexvar_ps(half, _mm512_loadu_ps(sf[r] + 2 * c)), acc[r]);
  }
  for (int r = 0; r < R; ++r) out[r] = _mm512_reduce_add_ps(acc[r]) + bsum[r];
}
int main() {
  const int K = 2560, NG = K / 64, ROWS = 64;  // 64 rows x 1280 B = 80 KB: L2-hot
  std::mt19937 rng(1);
  std::vector<uint8_t> w(ROWS * K / 2); for (auto& v : w) v = rng();
  std::vector<uint16_t> s(ROWS * NG), b(ROWS * NG), x(K);
  std::normal_distribution<float> nd(0, 1);
  for (auto& v : s) v = f2bf(nd(rng) * 0.01f); for (auto& v : b) v = f2bf(nd(rng) * 0.01f); for (auto& v : x) v = f2bf(nd(rng));
  std::vector<float> gs(NG, 0.f);
  for (int k = 0; k < K; ++k) gs[k / 64] += bf2f(x[k]);
  std::vector<uint32_t> xp(K / 2);
  for (int c = 0; c < K / 128; ++c) for (int k = 0; k < 4; ++k) for (int i = 0; i < 16; ++i)
    xp[c * 64 + k * 16 + i] = (uint32_t)x[c * 128 + 8 * i + 2 * k] | ((uint32_t)x[c * 128 + 8 * i + 2 * k + 1] << 16);
  std::vector<uint32_t> xq(K / 2);
  for (int c = 0; c < K / 128; ++c) for (int k = 0; k < 4; ++k) for (int i = 0; i < 16; ++i)
    xq[c * 64 + k * 16 + i] = (uint32_t)x[c * 128 + 8 * i + k] | ((uint32_t)x[c * 128 + 8 * i + k + 4] << 16);
  Lut lut;
  // reference in double
  double maxerr = 0;
  for (int r = 0; r < ROWS; ++r) {
    double ref = 0;
    for (int k = 0; k < K; ++k) {
      int code = (w[r * K / 2 + k / 2] >> (4 * (k & 1))) & 15;
      ref += (code * bf2f(s[r * NG + k / 64]) + bf2f(b[r * NG + k / 64])) * bf2f(x[k]);
    }
    float a = dotA(&w[r * K / 2], &s[r * NG], &b[r * NG], x.data(), gs.data(), K, lut.v);
    float bb = dotB<NG>(&w[r * K / 2], &s[r * NG], &b[r * NG], xp.data(), gs.data());
    if (r < 3) printf("row %d ref %.4f A %.4f B %.4f\n", r, ref, a, bb);
  }
  printf("max rel err vs fp64: %.2e\n", maxerr);
  {
    float o[4];
    dotC<NG, 4>(&w[0], K / 2, &s[0], &b[0], NG, xp.data(), gs.data(), o);
    printf("C rows 0-2: %.4f %.4f %.4f\n", o[0], o[1], o[2]);
  }
  {
    float o[4];
    dotD<NG, 4>(&w[0], K / 2, &s[0], &b[0], NG, xq.data(), gs.data(), o);
    printf("D rows 0-2: %.4f %.4f %.4f\n", o[0], o[1], o[2]);
    for (int R : {1, 2, 4}) {
      volatile float sink = 0; float oo[8];
      const int IT = 20000;
      auto t0 = std::chrono::steady_clock::now();
      for (int it = 0; it < IT; ++it)
        for (int r = 0; r < ROWS; r += R) {
          if (R == 1) dotD<NG, 1>(&w[r * K / 2], K / 2, &s[r * NG], &b[r * NG], NG, xq.data(), gs.data(), oo);
          else if (R == 2) dotD<NG, 2>(&w[r * K / 2], K / 2, &s[r * NG], &b[r * NG], NG, xq.data(), gs.data(), oo);
          else dotD<NG, 4>(&w[r * K / 2], K / 2, &s[r * NG], &b[r * NG], NG, xq.data(), gs.data(), oo);
          sink = sink + oo[0];
        }
      auto t1 = std::chrono::steady_clock::now();
      double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / ((double)IT * ROWS);
      printf("D %d rows: %.1f ns/row, %.1f GB/s per core\n", R, ns, (K / 2) / ns);
    }
  }
  for (int R : {2, 4, 8}) {
    volatile float sink = 0; float o[8];
    const int IT = 20000;
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < IT; ++it)
      for (int r = 0; r < ROWS; r += R) {
        if (R == 2) dotC<NG, 2>(&w[r * K / 2], K / 2, &s[r * NG], &b[r * NG], NG, xp.data(), gs.data(), o);
        else if (R == 4) dotC<NG, 4>(&w[r * K / 2], K / 2, &s[r * NG], &b[r * NG], NG, xp.data(), gs.data(), o);
        else dotC<NG, 8>(&w[r * K / 2], K / 2, &s[r * NG], &b[r * NG], NG, xp.data(), gs.data(), o);
        sink = sink + o[0];
      }
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / ((double)IT * ROWS);
    printf("C %d rows: %.1f ns/row, %.1f GB/s per core\n", R, ns, (K / 2) / ns);
  }
  for (int variant = 0; variant < 2; ++variant) {
    volatile float sink = 0;
    const int IT = 20000;
    auto t0 = std::chrono::steady_clock::now(); uint64_t c0 = __rdtsc();
    for (int it = 0; it < IT; ++it)
      for (int r = 0; r < ROWS; ++r)
        sink = sink + (variant == 0 ? dotA(&w[r * K / 2], &s[r * NG], &b[r * NG], x.data(), gs.data(), K, lut.v)
                                    : dotB<NG>(&w[r * K / 2], &s[r * NG], &b[r * NG], xp.data(), gs.data()));
    uint64_t c1 = __rdtsc(); auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / ((double)IT * ROWS);
    printf("%s: %.1f ns/row (%.1f tsc/group), %.1f GB/s per core of packed codes\n", variant ? "B magic " : "A vpermw", ns,
           (double)(c1 - c0) / ((double)IT * ROWS * NG), (K / 2) / ns);
  }
}
