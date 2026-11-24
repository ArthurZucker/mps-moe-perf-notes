// Host compute tier for streamed experts: experts that are not in the GPU pool are computed by CPU worker threads
// straight from their (pinned) host banks, instead of being read over PCIe. One token per call. The GPU posts the
// missed routes and the activation into a mapped mailbox (host_post in moe_affine.cu) and waits on the done flag.
#include <torch/all.h>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <sched.h>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>

namespace {

struct alignas(64) Mailbox {
  volatile uint64_t seq;  // written by the GPU last (after __threadfence_system)
  int32_t layer, n, ntok;
  int32_t ids[64];
  float w[64];
  int32_t tok[64];
  uint16_t x[4 * 8192];  // bf16 activation rows [ntok][H]
};
struct alignas(64) Result {
  volatile uint64_t done;
  float out[4 * 8192];  // [ntok][H]
  uint64_t t_seen_to_gate_ns, t_gate_to_down_ns, t_down_to_done_ns, calls;  // diagnostics (thread 0), after out
};
inline uint64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

inline float bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
inline float round_bf16(float f) {  // round-to-nearest-even to bf16, back to float (matches torch's bf16 cast)
  uint32_t u; memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  u &= 0xFFFF0000u;
  float r; memcpy(&r, &u, 4); return r;
}

// x permuted per 64-code group: xe[g*64 + i] = x[g*64 + 2i] (i < 32), then xo = odd codes; plus per-group sums
struct PermX {
  std::vector<float> xe, xo, gs;
  void build(const float* x, int K, int G) {
    xe.resize(K / 2); xo.resize(K / 2); gs.assign(K / G, 0.f);
    for (int k = 0; k < K; k += 2) { xe[k / 2] = x[k]; xo[k / 2] = x[k + 1]; gs[k / G] += x[k] + x[k + 1]; }
  }
};

// dot(dequant(row), x) for one packed row: K codes (K/8 uint32 words), group 64, bf16 scales/biases
inline float row_dot_g64(const uint32_t* w, const uint16_t* s, const uint16_t* b, const PermX& px, int K) {
  __m512 acc = _mm512_setzero_ps();
  float bacc = 0.f;
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(w);
  for (int g = 0; g < K / 64; ++g) {
    const uint8_t* gb = bytes + g * 32;  // 64 codes: byte j holds codes 2j (low nibble), 2j+1 (high)
    __m128i v0 = _mm_loadu_si128((const __m128i*)gb), v1 = _mm_loadu_si128((const __m128i*)(gb + 16));
    __m512 lo0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_and_si128(v0, m4)));
    __m512 lo1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_and_si128(v1, m4)));
    __m512 hi0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(v0, 4), m4)));
    __m512 hi1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(v1, 4), m4)));
    const float* xe = px.xe.data() + g * 32;
    const float* xo = px.xo.data() + g * 32;
    __m512 d = _mm512_mul_ps(lo0, _mm512_loadu_ps(xe));
    d = _mm512_fmadd_ps(lo1, _mm512_loadu_ps(xe + 16), d);
    d = _mm512_fmadd_ps(hi0, _mm512_loadu_ps(xo), d);
    d = _mm512_fmadd_ps(hi1, _mm512_loadu_ps(xo + 16), d);
    acc = _mm512_fmadd_ps(d, _mm512_set1_ps(bf2f(s[g])), acc);
    bacc += bf2f(b[g]) * px.gs[g];
  }
  return _mm512_reduce_add_ps(acc) + bacc;
}

struct LayerBanks { const uint32_t* gw; const uint16_t* gs; const uint16_t* gb; const uint32_t* dw; const uint16_t* ds; const uint16_t* db; };

struct Tier {
  int H, I, G, nthreads;
  std::vector<LayerBanks> layers;
  Mailbox* mb = nullptr;
  Result* res = nullptr;
  std::vector<std::thread> threads;
  std::atomic<bool> stop{false};
  // phase barrier
  std::atomic<int> arrived{0};
  std::atomic<int> generation{0};
  std::vector<float> gate_up;  // [n][2I] bf16-rounded
  std::vector<float> act_shared;  // [n][I]
  std::vector<float> outbuf;   // [H]
  uint64_t calls = 0;

  void barrier() {
    int gen = generation.load(std::memory_order_acquire);
    if (arrived.fetch_add(1, std::memory_order_acq_rel) == nthreads - 1) {
      arrived.store(0, std::memory_order_relaxed);
      generation.fetch_add(1, std::memory_order_acq_rel);
    } else {
      while (generation.load(std::memory_order_acquire) == gen) _mm_pause();
    }
  }

  void worker(int t, int cpu) {
    if (cpu >= 0) { cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set); sched_setaffinity(0, sizeof(set), &set); }
    uint64_t last = 0;
    PermX px, pa;
    std::vector<float> xf(H), act;
    while (true) {
      uint64_t s;
      while ((s = mb->seq) == last) {
        if (stop.load(std::memory_order_relaxed)) return;
        _mm_pause();
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      last = s;
      const uint64_t t0 = t == 0 ? now_ns() : 0;
      const int L = mb->layer, n = mb->n, ntok = mb->ntok;
      const LayerBanks& lb = layers[L];
      std::vector<PermX> pxs(ntok);
      for (int q = 0; q < ntok; ++q) {
        for (int k = 0; k < H; ++k) xf[k] = bf2f(mb->x[(size_t)q * H + k]);
        pxs[q].build(xf.data(), H, G);
      }
      // phase 1: gate/up rows of every missed expert
      const int rows1 = n * 2 * I;
      const int c1 = (rows1 + nthreads - 1) / nthreads;  // contiguous row blocks: sequential reads per thread
      for (int r = t * c1; r < std::min(rows1, (t + 1) * c1); ++r) {
        const int j = r / (2 * I), row = r % (2 * I);
        const long e = mb->ids[j];
        const long wrow = (e * 2 * I + row);
        gate_up[r] = round_bf16(row_dot_g64(lb.gw + wrow * (H / 8), lb.gs + wrow * (H / G), lb.gb + wrow * (H / G), pxs[mb->tok[j]], H));
      }
      barrier();
      const uint64_t t1 = t == 0 ? now_ns() : 0;
      // act = bf16(bf16(silu(gate)) * up), each thread a slice into the shared buffer, then every thread permutes it
      {
        const int total = n * I, ca = (total + nthreads - 1) / nthreads;
        for (int i = t * ca; i < std::min(total, (t + 1) * ca); ++i) {
          const int j = i / I, k = i - j * I;
          const float g = gate_up[j * 2 * I + k], u = gate_up[j * 2 * I + I + k];
          act_shared[i] = round_bf16(round_bf16(g / (1.f + std::exp(-g))) * u);
        }
      }
      barrier();
      std::vector<PermX> pas(n);
      for (int j = 0; j < n; ++j) pas[j].build(act_shared.data() + (size_t)j * I, I, G);
      const int c2 = (H + nthreads - 1) / nthreads;
      const int r0 = t * c2, r1 = std::min(H, (t + 1) * c2);
      // experts outer, rows inner: each expert's down rows stream sequentially; partial sums per row
      for (int q = 0; q < ntok; ++q)
        for (int r = r0; r < r1; ++r) res->out[(size_t)q * H + r] = 0.f;
      for (int j = 0; j < n; ++j) {
        const long e = mb->ids[j];
        const float wj = mb->w[j];
        float* outq = res->out + (size_t)mb->tok[j] * H;
        for (int r = r0; r < r1; ++r) {
          const long wrow = e * H + r;
          outq[r] += wj * row_dot_g64(lb.dw + wrow * (I / 8), lb.ds + wrow * (I / G), lb.db + wrow * (I / G), pas[j], I);
        }
      }
      barrier();
      if (t == 0) {
        const uint64_t t2 = now_ns();
        res->t_seen_to_gate_ns += t1 - t0; res->t_gate_to_down_ns += t2 - t1; res->calls += 1;
        std::atomic_thread_fence(std::memory_order_release);
        _mm_sfence();
        res->done = s;
        ++calls;
      }
    }
  }
};

std::vector<Tier*> g_tiers;

}  // namespace

// banks: per layer 6 host pointers (gate_up w/s/b, down w/s/b); cpus: core ids to pin the workers to (may be empty)
// returns (handle, mailbox address, result address); both buffers are mapped pinned memory (UVA: same address on the GPU)
std::tuple<int64_t, int64_t, int64_t> host_tier_start(std::vector<int64_t> banks, int64_t H, int64_t I, int64_t G,
                                                      std::vector<int64_t> cpus) {
  TORCH_CHECK(G == 64 && H <= 8192 && banks.size() % 6 == 0, "host tier: group 64, H <= 8192");
  auto* tier = new Tier();
  tier->H = H; tier->I = I; tier->G = G; tier->nthreads = (int)cpus.size();
  TORCH_CHECK(tier->nthreads >= 1, "host tier: need at least one cpu");
  for (size_t i = 0; i < banks.size(); i += 6)
    tier->layers.push_back({(const uint32_t*)banks[i], (const uint16_t*)banks[i + 1], (const uint16_t*)banks[i + 2],
                            (const uint32_t*)banks[i + 3], (const uint16_t*)banks[i + 4], (const uint16_t*)banks[i + 5]});
  TORCH_CHECK(cudaHostAlloc((void**)&tier->mb, sizeof(Mailbox), cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess);
  TORCH_CHECK(cudaHostAlloc((void**)&tier->res, sizeof(Result), cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess);
  memset((void*)tier->mb, 0, sizeof(Mailbox));
  memset((void*)tier->res, 0, sizeof(Result));
  tier->gate_up.resize(64 * 2 * I);
  tier->act_shared.resize(64 * I);
  for (int t = 0; t < tier->nthreads; ++t) tier->threads.emplace_back(&Tier::worker, tier, t, (int)cpus[t]);
  g_tiers.push_back(tier);
  return std::make_tuple((int64_t)(g_tiers.size() - 1), (int64_t)tier->mb, (int64_t)tier->res);
}

int64_t host_tier_calls(int64_t handle) { return (int64_t)g_tiers.at(handle)->calls; }

std::vector<int64_t> host_tier_timing(int64_t handle) {
  Result* r = g_tiers.at(handle)->res;
  std::vector<int64_t> v{(int64_t)r->t_seen_to_gate_ns, (int64_t)r->t_gate_to_down_ns, (int64_t)r->calls};
  r->t_seen_to_gate_ns = r->t_gate_to_down_ns = r->calls = 0;
  return v;
}

void host_tier_stop(int64_t handle) {
  Tier* tier = g_tiers.at(handle);
  if (!tier) return;
  tier->stop = true;
  for (auto& th : tier->threads) th.join();
  cudaFreeHost(tier->mb); cudaFreeHost(tier->res);
  delete tier;
  g_tiers[handle] = nullptr;
}
