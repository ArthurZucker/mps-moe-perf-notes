// Host compute tier v2: missed experts computed by CPU workers straight from their pinned host banks.
// vs v1: codes expanded to bf16 by one vpermw table lookup and multiplied with vdpbf16ps (32 MACs per instruction)
// against the bf16 activation in natural order (no per-call permutation); the GPU posts the activation's per-group
// sums with it; no per-call allocations; each thread prefetches its down rows into L2 while the gate phase runs.
#include <torch/all.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>
#include <algorithm>

namespace {

constexpr int MAXR = 64, MAXT = 4, MAXH = 8192, RING = 1024;
struct alignas(64) Mailbox2 {
  // line 0: everything a waiting worker watches (umonitor covers one line): post and hint sequence numbers
  volatile uint64_t seq;       // posts (GPU)
  volatile uint64_t hint_seq;  // prefetch hints (GPU)
  int32_t layer, n, ntok, hint_layer, hint_n, pad[5];
  int32_t ids[MAXR];
  float w[MAXR];
  int32_t tok[MAXR];
  int32_t hint_ids[MAXR];
  alignas(64) uint32_t xq[MAXT * MAXH / 2];   // bf16 rows [ntok][H] in pair layout (see dot_rows)
  alignas(64) float xgs[MAXT * (MAXH / 64)];  // their per-64-group sums
  // pool installs (GPU plans, the CPU installer copies with the copy engine, the GPU commits): ring of (key, slot)
  alignas(64) volatile uint64_t inst_head;     // entries produced (GPU)
  alignas(64) volatile uint64_t inst_done;     // entries copied (CPU installer), in order
  alignas(64) int32_t ring[RING][2];
};
static_assert(offsetof(Mailbox2, hint_n) < 64, "watched fields share line 0");
struct alignas(64) Result2 {
  volatile uint64_t done;
  uint64_t pad[7];
  alignas(64) float out[MAXT * MAXH];  // [ntok][H]
  uint64_t t_gate_ns, t_act_ns, t_down_ns, calls;
  uint64_t bad_posts, bad_hints, hints, routes;
};
static_assert(offsetof(Mailbox2, xq) % 64 == 0 && offsetof(Result2, out) == 64, "layout shared with moe_affine.cu");

// wait in C0.1 until the line of `p` is written or ~`cycles` TSC ticks pass (WAITPKG): no spinning power draw
inline void wait_line(const volatile void* p, uint64_t cycles) {
  _umonitor((void*)p);
  _umwait(1, __rdtsc() + cycles);
}

inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline float bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
inline uint16_t f2bf(float f) {  // round to nearest even (torch's bf16 cast)
  uint32_t u; memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}


// R rows of K packed codes (group 64) against one activation in pair layout xq: per 128-code chunk c, uint32
// xq[c*64 + k*16 + i] = bf16 pair (x[c*128 + 8i + k], x[c*128 + 8i + k + 4]), matching lane i of the chunk's 16 words
// decoded as ((w >> 4k) & 0x000F000F) | 0x43004300 = bf16 pair (128 + nibble k, 128 + nibble k + 4).
// row value = sum_g s_g * D_g + (b_g - 128 s_g) * gsum_g, D_g = sum over the group of (128 + c) x.
template <int R>
inline void dot_rows(const uint8_t* __restrict__ wb, size_t row_bytes, const uint16_t* __restrict__ s,
                     const uint16_t* __restrict__ b, size_t sstride, const uint32_t* __restrict__ xq,
                     const float* __restrict__ gs, int NG, float* out) {
  const __m512i C = _mm512_set1_epi32(0x43004300), M = _mm512_set1_epi32(0x000F000F);
  const __m512i half = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
  alignas(64) float sf[R][MAXH / 64 + 16];
  __m512 bsum[R];
  for (int r = 0; r < R; ++r) {  // scales -> fp32, and the (b - 128 s) . gsum term, 16 groups per step
    bsum[r] = _mm512_setzero_ps();
    for (int g = 0; g < NG; g += 16) {
      const __mmask16 m = NG - g >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << (NG - g)) - 1);
      const __m512 sv = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(m, s + r * sstride + g)), 16));
      const __m512 bv = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(m, b + r * sstride + g)), 16));
      _mm512_store_ps(sf[r] + g, sv);
      bsum[r] = _mm512_fmadd_ps(_mm512_fnmadd_ps(_mm512_set1_ps(128.f), sv, bv), _mm512_maskz_loadu_ps(m, gs + g), bsum[r]);
    }
  }
  __m512 acc[R];
#pragma unroll
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
#pragma unroll
  for (int r = 0; r < R; ++r) out[r] = _mm512_reduce_add_ps(_mm512_add_ps(acc[r], bsum[r]));
}

// rows [r0, r1) of one expert bank (row-major, `row_bytes` packed bytes, `ng` groups per row)
inline void dot_range(const uint8_t* wb, size_t row_bytes, const uint16_t* s, const uint16_t* b, int ng, int r0, int r1,
                      const uint32_t* xq, const float* gs, float* out) {
  int r = r0;
  for (; r + 4 <= r1; r += 4) dot_rows<4>(wb + r * row_bytes, row_bytes, s + (size_t)r * ng, b + (size_t)r * ng, ng, xq, gs, ng, out + (r - r0));
  for (; r < r1; ++r) dot_rows<1>(wb + r * row_bytes, row_bytes, s + (size_t)r * ng, b + (size_t)r * ng, ng, xq, gs, ng, out + (r - r0));
}

// read every 64-byte line of [p, p + bytes) (lands in L1/L2): returns a value the caller keeps
inline uint64_t touch(const void* p, size_t bytes) {
  const char* c = (const char*)p;
  __m512i acc = _mm512_setzero_si512();
  size_t o = 0;
  for (; o + 64 <= bytes; o += 64) acc = _mm512_xor_si512(acc, _mm512_loadu_si512(c + o));
  uint64_t v = (uint64_t)_mm512_reduce_or_epi64(acc);
  for (; o < bytes; o += 8) v ^= (uint64_t)c[o];
  return v;
}

struct LayerBanks { const uint8_t* gw; const uint16_t* gs; const uint16_t* gb; const uint8_t* dw; const uint16_t* ds; const uint16_t* db; };

struct Tier2 {
  int H, I, G, T;
  std::vector<LayerBanks> layers;     // primary banks (pinned, the GPU installs from them)
  std::vector<LayerBanks> replica;    // the same banks in `replica_node`'s memory (empty: none)
  int replica_node = -1;
  std::vector<int> thread_node;       // NUMA node of each worker's cpu
  Mailbox2* mb = nullptr;
  Result2* res = nullptr;
  std::vector<std::thread> threads;
  // installer: copies planned (key, slot) installs host bank -> pool with cudaMemcpyAsync
  std::thread installer_thread;
  void* pool[6] = {};
  size_t expert_bytes[6] = {};
  int E = 0;
  std::atomic<uint64_t> installs{0};
  // Quiet toward the driver (its locks are shared with the thread launching the decode graphs): pending installs are
  // issued as one cudaMemcpyBatchAsync at most every `pace_us`, and the stream itself writes inst_done into the mapped
  // mailbox after them (cuStreamWriteValue64): no per-copy calls, no synchronization calls.
  void installer(int device, int cpu) {
    if (cpu >= 0) { cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set); sched_setaffinity(0, sizeof(set), &set); }
    cudaSetDevice(device);
    using WriteValue64 = CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
    WriteValue64 write_value = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPointByVersion("cuStreamWriteValue64", (void**)&write_value, 12000, cudaEnableDefault, &q);
    TORCH_CHECK(write_value && q == cudaDriverEntryPointSuccess, "installer: cuStreamWriteValue64 unavailable");
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    std::vector<void*> dst; std::vector<const void*> src; std::vector<size_t> size;
    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    size_t attr_idx = 0;
    uint64_t issued = mb->inst_done;
    const char* pace_env = getenv("HOST_INSTALL_PACE_US");
    const uint64_t pace_ns = (pace_env ? strtoull(pace_env, nullptr, 10) : 300) * 1000;
    uint64_t last_issue = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      const uint64_t head = mb->inst_head;
      if (head == issued || now_ns() - last_issue < pace_ns) {
        _umonitor((void*)&mb->inst_head);
        if (mb->inst_head == issued) _umwait(1, __rdtsc() + 200000);
        else _mm_pause();
        continue;
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      dst.clear(); src.clear(); size.clear();
      for (uint64_t i = issued; i < head; ++i) {
        const int key = mb->ring[i % RING][0], slot = mb->ring[i % RING][1];
        const int L = key / E, e = key % E;
        if (L < 0 || L >= (int)layers.size() || e < 0 || slot < 0) continue;
        const LayerBanks& lb = layers[L];
        const void* srcs[6] = {lb.gw, lb.gs, lb.gb, lb.dw, lb.ds, lb.db};
        for (int b = 0; b < 6; ++b) {
          dst.push_back((char*)pool[b] + (size_t)slot * expert_bytes[b]);
          src.push_back((const char*)srcs[b] + (size_t)e * expert_bytes[b]);
          size.push_back(expert_bytes[b]);
        }
      }
      if (!dst.empty())
        cudaMemcpyBatchAsync(dst.data(), src.data(), size.data(), dst.size(), &attr, &attr_idx, 1, st);
      write_value((CUstream)st, (CUdeviceptr)&mb->inst_done, (cuuint64_t)head, 0);
      installs.fetch_add(head - issued, std::memory_order_relaxed);
      issued = head;
      last_issue = now_ns();
    }
    cudaStreamSynchronize(st);
    cudaStreamDestroy(st);
  }
  std::atomic<bool> stop{false};
  std::vector<uint16_t> gate_up;  // unused (kept for layout of older diagnostics)
  std::vector<uint16_t> act;      // [n][I] bf16 elements of the pair layout (uint16 view)
  std::vector<float> ags;         // unused
  uint64_t calls = 0;
  bool umwait = false;  // idle workers wait in C0.1 (HOST_UMWAIT=1): less power, ~7 us later to start a call
  // dissemination barrier: round r, thread t signals (t + 2^r) % T and waits for (t - 2^r) % T; per-thread epochs
  struct alignas(64) Flag { std::atomic<uint64_t> v{0}; };
  std::vector<Flag> flags;  // [rounds][T]
  int rounds = 0;
  void barrier(int t, uint64_t& epoch) {
    ++epoch;
    for (int r = 0; r < rounds; ++r) {
      const int to = (t + (1 << r)) % T;
      flags[(size_t)r * T + to].v.store(epoch, std::memory_order_release);
      auto& mine = flags[(size_t)r * T + t].v;
      while (mine.load(std::memory_order_acquire) < epoch) _mm_pause();  // short waits: spin (umwait wakeups cost ~1 us)
    }
  }

  void worker(int t, int cpu) {
    if (cpu >= 0) { cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set); sched_setaffinity(0, sizeof(set), &set); }
    // a worker reads the banks in its own node's memory
    const std::vector<LayerBanks>& banks = (!replica.empty() && thread_node[t] == replica_node) ? replica : layers;
    std::vector<float> outacc((size_t)MAXT * ((H + T - 1) / T));
    uint64_t last = 0, last_hint = 0;
    uint64_t sink = 0;
    const int rb_g = H / 2, rb_d = I / 2;  // packed bytes per row
    // fixed row slices of every expert: gate/up rows [g0, g1) of 2I, down rows [r0, r1) of H (prefetchable before n is known)
    // gate/up: thread t owns k in [k0, k1) of both halves (gate rows k, up rows I + k): its act slice needs no barrier
    const int ck = ((I + T - 1) / T + 3) / 4 * 4, k0 = std::min(I, t * ck), k1 = std::min(I, k0 + ck);
    const int g0 = k0, g1 = k1;  // hint staging: gate rows [k0, k1) and up rows [I + k0, I + k1)
    const int cd = ((H + T - 1) / T + 3) / 4 * 4, r0 = std::min(H, t * cd), r1 = std::min(H, r0 + cd);
    std::vector<float> rowbuf(std::max(2 * ck, cd));
    std::vector<float> agsl((size_t)MAXR * (I / 64));
    uint64_t epoch = 0;
    while (true) {
      uint64_t s;
      while ((s = mb->seq) == last) {
        if (stop.load(std::memory_order_relaxed)) return;
        const uint64_t h = mb->hint_seq;
        if (h != last_hint) {  // stage this thread's slices of the hinted experts into L2 while the GPU runs attention
          last_hint = h;
          std::atomic_thread_fence(std::memory_order_acquire);
          const int HL = mb->hint_layer, hn = std::min<int>(mb->hint_n, MAXR);
          if (HL < 0 || HL >= (int)layers.size()) continue;
          const LayerBanks& hb = banks[HL];
          for (int j = 0; j < hn && mb->seq == last; ++j) {
            const int he = mb->hint_ids[j];
            if (he < 0 || he >= E) { if (t == 0) res->bad_hints += 1; continue; }  // never trust an index from the mailbox
            if (t == 0 && j == 0) res->hints += 1;
            const size_t e = (size_t)he;
            sink ^= touch(hb.gw + (e * 2 * I + g0) * rb_g, (size_t)(g1 - g0) * rb_g);
            sink ^= touch(hb.gw + (e * 2 * I + I + g0) * rb_g, (size_t)(g1 - g0) * rb_g);
            sink ^= touch(hb.gs + (e * 2 * I + I + g0) * (H / 64), (size_t)(g1 - g0) * (H / 64) * 2);
            sink ^= touch(hb.gb + (e * 2 * I + I + g0) * (H / 64), (size_t)(g1 - g0) * (H / 64) * 2);
            sink ^= touch(hb.dw + (e * H + r0) * rb_d, (size_t)(r1 - r0) * rb_d);
            sink ^= touch(hb.gs + (e * 2 * I + g0) * (H / 64), (size_t)(g1 - g0) * (H / 64) * 2);
            sink ^= touch(hb.gb + (e * 2 * I + g0) * (H / 64), (size_t)(g1 - g0) * (H / 64) * 2);
            sink ^= touch(hb.ds + (e * H + r0) * (I / 64), (size_t)(r1 - r0) * (I / 64) * 2);
            sink ^= touch(hb.db + (e * H + r0) * (I / 64), (size_t)(r1 - r0) * (I / 64) * 2);
          }
          continue;
        }
        if (!umwait) { _mm_pause(); continue; }
        if (mb->seq == last && mb->hint_seq == last_hint) {
          _umonitor((void*)&mb->seq);
          if (mb->seq == last && mb->hint_seq == last_hint) _umwait(1, __rdtsc() + 200000);
        }
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      last = s;
      const uint64_t t0 = t == 0 ? now_ns() : 0;
      const int L = mb->layer, ntok = mb->ntok;
      int n = mb->n;
      bool valid = L >= 0 && L < (int)layers.size() && n >= 0 && n <= MAXR && ntok >= 1 && ntok <= MAXT;
      for (int j = 0; valid && j < n; ++j) valid = mb->ids[j] >= 0 && mb->ids[j] < E && mb->tok[j] >= 0 && mb->tok[j] < ntok;
      if (!valid) { n = 0; if (t == 0) res->bad_posts += 1; }  // a malformed post: answer zeros rather than read out of bounds
      if (t == 0) res->routes += n;
      const LayerBanks& lb = banks[std::max(0, std::min(L, (int)banks.size() - 1))];
      // this thread's down rows: prefetch into L2 now, they are read after the gate phase
      for (int j = 0; j < n; ++j) {
        const uint8_t* p = lb.dw + ((size_t)mb->ids[j] * H + r0) * rb_d;
        const size_t bytes = (size_t)(r1 - r0) * rb_d;
        for (size_t o = 0; o < bytes; o += 64) _mm_prefetch((const char*)p + o, _MM_HINT_T1);
      }
      // phase 1: gate rows [k0, k1) and up rows [I + k0, I + k1) of every missed expert -> act slice (pair layout)
      uint16_t* act16 = act.data();
      for (int j = 0; j < n; ++j) {
        const int q = mb->tok[j];
        const size_t e = mb->ids[j];
        const uint8_t* gw = lb.gw + e * 2 * I * rb_g;
        const uint16_t* gsc = lb.gs + e * 2 * I * (H / 64);
        const uint16_t* gbi = lb.gb + e * 2 * I * (H / 64);
        const uint32_t* xq = mb->xq + (size_t)q * H / 2;
        const float* xg = mb->xgs + (size_t)q * (H / 64);
        dot_range(gw, rb_g, gsc, gbi, H / 64, k0, k1, xq, xg, rowbuf.data());
        dot_range(gw, rb_g, gsc, gbi, H / 64, I + k0, I + k1, xq, xg, rowbuf.data() + (k1 - k0));
        for (int k = k0; k < k1; ++k) {
          const float gv = bf2f(f2bf(rowbuf[k - k0])), u = bf2f(f2bf(rowbuf[(k1 - k0) + k - k0]));
          const uint16_t a = f2bf(bf2f(f2bf(gv / (1.f + std::exp(-gv)))) * u);
          // element k of the pair layout: chunk k / 128, p = k % 128, lane i = p / 8, nibble slot p % 8
          const int c = k / 128, p = k % 128, i = p / 8, kk = p % 8;
          act16[(size_t)j * I + 2 * (c * 64 + (kk & 3) * 16 + i) + (kk >> 2)] = a;
        }
      }
      barrier(t, epoch);
      const uint64_t t1 = t == 0 ? now_ns() : 0;
      // group sums of each act (cheap: every thread computes them all)
      for (int j = 0; j < n; ++j)
        for (int g = 0; g < I / 64; ++g) {
          // group g = lanes [8 (g % 2), 8 (g % 2) + 8) of chunk g / 2, all 4 k slots, both halves
          const uint32_t* aq = reinterpret_cast<const uint32_t*>(act16 + (size_t)j * I) + (g / 2) * 64 + (g % 2) * 8;
          float sum = 0.f;
          for (int kk = 0; kk < 4; ++kk)
            for (int ii = 0; ii < 8; ++ii) { const uint32_t v = aq[kk * 16 + ii]; sum += bf2f(v & 0xFFFF) + bf2f(v >> 16); }
          agsl[(size_t)j * (I / 64) + g] = sum;
        }
      const uint64_t t2 = t == 0 ? now_ns() : 0;
      // phase 3: down rows r0..r1, experts outer (each expert's rows stream sequentially), routing-weighted sum
      std::fill(outacc.begin(), outacc.begin() + (size_t)ntok * (r1 - r0), 0.f);
      for (int j = 0; j < n; ++j) {
        const size_t e = mb->ids[j];
        const float wj = mb->w[j];
        float* oq = outacc.data() + (size_t)mb->tok[j] * (r1 - r0);
        dot_range(lb.dw + e * H * rb_d, rb_d, lb.ds + e * H * (I / 64), lb.db + e * H * (I / 64), I / 64, r0, r1,
                  reinterpret_cast<const uint32_t*>(act16 + (size_t)j * I), agsl.data() + (size_t)j * (I / 64), rowbuf.data());
        for (int r = r0; r < r1; ++r) oq[r - r0] += wj * rowbuf[r - r0];
      }
      for (int q = 0; q < ntok; ++q) memcpy(res->out + (size_t)q * H + r0, outacc.data() + (size_t)q * (r1 - r0), (r1 - r0) * 4);
      barrier(t, epoch);
      if (sink == 0x123456789ull) res->pad[0] = sink;  // keep the touches
      if (t == 0) {
        const uint64_t t3 = now_ns();
        res->t_gate_ns += t1 - t0; res->t_act_ns += t2 - t1; res->t_down_ns += t3 - t2; res->calls += 1;
        std::atomic_thread_fence(std::memory_order_release);
        _mm_sfence();
        res->done = s;
        ++calls;
      }
    }
  }
};

std::vector<Tier2*> g_tiers2;

}  // namespace

// copy `bytes` at `src` into fresh memory bound to NUMA node `node` (transparent huge pages), with `nt` threads
static void* replicate(const void* src, size_t bytes, int node, int nt) {
  const size_t len = (bytes + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
  void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  TORCH_CHECK(p != MAP_FAILED, "replica: mmap");
  madvise(p, len, MADV_HUGEPAGE);
  unsigned long mask = 1ul << node;
  TORCH_CHECK(syscall(SYS_mbind, p, len, 2 /* MPOL_BIND */, &mask, sizeof(mask) * 8, 0) == 0, "replica: mbind");
  std::vector<std::thread> th;
  const size_t chunk = (bytes + nt - 1) / nt;
  for (int i = 0; i < nt; ++i)
    th.emplace_back([=] {
      const size_t a = std::min(bytes, i * chunk), b = std::min(bytes, a + chunk);
      if (b > a) memcpy((char*)p + a, (const char*)src + a, b - a);
    });
  for (auto& x : th) x.join();
  return p;
}

std::tuple<int64_t, int64_t, int64_t> host_tier2_start(std::vector<int64_t> banks, int64_t H, int64_t I, int64_t G,
                                                       std::vector<int64_t> cpus, int64_t E) {
  TORCH_CHECK(G == 64 && H <= MAXH && H % 128 == 0 && I % 128 == 0 && banks.size() % 6 == 0, "host tier 2: group 64");
  auto* tier = new Tier2();
  tier->H = H; tier->I = I; tier->G = G; tier->T = (int)cpus.size(); tier->E = (int)E;
  const char* uw = getenv("HOST_UMWAIT");
  tier->umwait = uw && uw[0] == '1';
  TORCH_CHECK(E > 0, "host tier 2: number of experts per layer");
  TORCH_CHECK(tier->T >= 1, "host tier 2: need at least one cpu");
  for (size_t i = 0; i < banks.size(); i += 6)
    tier->layers.push_back({(const uint8_t*)banks[i], (const uint16_t*)banks[i + 1], (const uint16_t*)banks[i + 2],
                            (const uint8_t*)banks[i + 3], (const uint16_t*)banks[i + 4], (const uint16_t*)banks[i + 5]});
  TORCH_CHECK(cudaHostAlloc((void**)&tier->mb, sizeof(Mailbox2), cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess);
  TORCH_CHECK(cudaHostAlloc((void**)&tier->res, sizeof(Result2), cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess);
  memset((void*)tier->mb, 0, sizeof(Mailbox2));
  memset((void*)tier->res, 0, sizeof(Result2));
  tier->act.resize((size_t)MAXR * I);
  // optional replica of the banks on a second NUMA node (HOST_REPLICA_NODE), for the workers on that node
  // (HOST_CPU_NODES: comma-separated node of each worker cpu, same order as `cpus`)
  tier->thread_node.assign(tier->T, -1);
  if (const char* cn = getenv("HOST_CPU_NODES")) {
    std::vector<int> nodes; const char* q = cn;
    while (*q) { nodes.push_back(atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
    TORCH_CHECK((int)nodes.size() == tier->T, "HOST_CPU_NODES: one node per worker cpu");
    tier->thread_node = nodes;
  }
  if (const char* rn = getenv("HOST_REPLICA_NODE")) {
    tier->replica_node = atoi(rn);
    const size_t gw = (size_t)E * 2 * I * H / 2, gs = (size_t)E * 2 * I * (H / 64) * 2, dw = (size_t)E * H * I / 2, ds = (size_t)E * H * (I / 64) * 2;
    for (const LayerBanks& lb : tier->layers)
      tier->replica.push_back({(const uint8_t*)replicate(lb.gw, gw, tier->replica_node, 16), (const uint16_t*)replicate(lb.gs, gs, tier->replica_node, 4),
                               (const uint16_t*)replicate(lb.gb, gs, tier->replica_node, 4), (const uint8_t*)replicate(lb.dw, dw, tier->replica_node, 16),
                               (const uint16_t*)replicate(lb.ds, ds, tier->replica_node, 4), (const uint16_t*)replicate(lb.db, ds, tier->replica_node, 4)});
  }
  while ((1 << tier->rounds) < tier->T) ++tier->rounds;
  tier->flags = std::vector<Tier2::Flag>((size_t)std::max(tier->rounds, 1) * tier->T);
  for (int t = 0; t < tier->T; ++t) tier->threads.emplace_back(&Tier2::worker, tier, t, (int)cpus[t]);
  g_tiers2.push_back(tier);
  return std::make_tuple((int64_t)(g_tiers2.size() - 1), (int64_t)tier->mb, (int64_t)tier->res);
}

std::vector<int64_t> host_tier2_timing(int64_t handle) {
  Result2* r = g_tiers2.at(handle)->res;
  std::vector<int64_t> v{(int64_t)r->t_gate_ns, (int64_t)r->t_act_ns, (int64_t)r->t_down_ns, (int64_t)r->bad_posts,
                         (int64_t)r->bad_hints, (int64_t)r->hints, (int64_t)r->routes, (int64_t)r->calls};
  r->t_gate_ns = r->t_act_ns = r->t_down_ns = r->calls = r->bad_posts = r->bad_hints = r->hints = r->routes = 0;
  return v;
}

// start the pool installer: `pool` = the 6 slot-indexed device banks (gate w/s/b, down w/s/b), E experts per layer
void host_tier2_installer(int64_t handle, std::vector<int64_t> pool, std::vector<int64_t> expert_bytes, int64_t E,
                          int64_t device, int64_t cpu) {
  Tier2* tier = g_tiers2.at(handle);
  TORCH_CHECK(pool.size() == 6 && expert_bytes.size() == 6 && !tier->installer_thread.joinable());
  for (int b = 0; b < 6; ++b) { tier->pool[b] = (void*)pool[b]; tier->expert_bytes[b] = (size_t)expert_bytes[b]; }
  TORCH_CHECK((int)E == tier->E, "installer: experts per layer differ from host_tier2_start");
  tier->installer_thread = std::thread(&Tier2::installer, tier, (int)device, (int)cpu);
}

// wait until every planned install has been copied (before the pool is written by other means, e.g. a prefill seed)
int64_t host_tier2_drain(int64_t handle) {
  Tier2* tier = g_tiers2.at(handle);
  if (!tier->installer_thread.joinable()) return 0;
  while (tier->mb->inst_done != tier->mb->inst_head) _mm_pause();
  return (int64_t)tier->installs.load();
}

void host_tier2_stop(int64_t handle) {
  Tier2* tier = g_tiers2.at(handle);
  if (!tier) return;
  tier->stop = true;
  for (auto& th : tier->threads) th.join();
  if (tier->installer_thread.joinable()) tier->installer_thread.join();
  cudaFreeHost(tier->mb); cudaFreeHost(tier->res);
  delete tier;
  g_tiers2[handle] = nullptr;
}
