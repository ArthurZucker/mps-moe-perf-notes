// Microbenchmark for PyTorch's MPS stream: what does encoding every kernel into one serial compute encoder cost, versus
// a concurrent encoder with explicit barriers? Runs on torch's MTLDevice, own command buffers, GPU timestamps.
#include <torch/extension.h>
#include <ATen/mps/MPSDevice.h>
#import <Metal/Metal.h>
#include <string>
#include <vector>
#include <chrono>

static const char* kSource = R"(
#include <metal_stdlib>
using namespace metal;
// tiny: one element per thread
kernel void tiny(device const float* a [[buffer(0)]], device float* b [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    b[i] = a[i] * 1.0001f + 0.5f;
}
// matvec: y[row] = W[row, :] . x, one simdgroup per row, K = 2304 bf16 (a decode-shaped projection, 1024 x 2304)
// touch: read a weight block into the cache (sum, write one value so it is not optimized away)
kernel void touch(device const uint4* w [[buffer(0)]], device float* sink [[buffer(1)]], uint i [[thread_position_in_grid]],
                  uint n [[threads_per_grid]]) {
    uint4 acc = 0;
    for (uint k = i; k < 1024 * 2304 * 2 / 16; k += n) acc ^= w[k];
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) sink[0] = 1.0f;
}
// software dependent launch: load this threadgroup's weight rows into registers, then wait for the previous step's
// threadgroups to finish (device counter), then read x and compute; finally publish completion.
kernel void matvec_pdl(device const bfloat* w [[buffer(0)]], coherent(device) device const float* x [[buffer(1)]],
                       coherent(device) device float* y [[buffer(2)]], device atomic_uint* done_prev [[buffer(3)]],
                       device atomic_uint* done_self [[buffer(4)]], constant uint& need [[buffer(5)]],
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    const uint row = tg * 8 + sg;
    float wr[72];
    for (uint t = 0; t < 9; ++t)
        for (uint j = 0; j < 8; ++j) wr[t * 8 + j] = float(w[row * 2304 + lane * 8 + t * 256 + j]);  // independent of x
    if (need > 0) {
        if (tid == 0) {
            for (uint spin = 0; spin < (1u << 22); ++spin)
                if (atomic_load_explicit(done_prev, memory_order_relaxed) >= need) break;
        }
        threadgroup_barrier(mem_flags::mem_device);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    float acc = 0.0f;
    for (uint t = 0; t < 9; ++t)
        for (uint j = 0; j < 8; ++j) acc += wr[t * 8 + j] * x[lane * 8 + t * 256 + j];
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        atomic_fetch_add_explicit(done_self, 1u, memory_order_relaxed);
    }
}
kernel void matvec(device const bfloat* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                   uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg * 8 + sg;
    float acc = 0.0f;
    for (uint k = lane * 8; k < 2304; k += 256)
        for (uint j = 0; j < 8; ++j) acc += float(w[row * 2304 + k + j]) * x[k + j];
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}
)";

struct State {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> tiny, matvec, touch, pdl;
  id<MTLBuffer> counters;
  id<MTLCommandQueue> side;
  std::vector<id<MTLBuffer>> weights;  // distinct weights per chain step (DRAM-bound ring)
  id<MTLBuffer> sink;
  std::vector<id<MTLBuffer>> act;  // activation ring (chain: dispatch i reads act[i], writes act[i + 1])
  id<MTLBuffer> weight;
};
static State* S = nullptr;

static void init() {
  if (S) return;
  S = new State();
  S->device = at::mps::MPSDevice::getInstance()->device();
  S->queue = [S->device newCommandQueue];
  NSError* err = nil;
  MTLCompileOptions* opts = [MTLCompileOptions new];
  opts.languageVersion = MTLLanguageVersion3_2;  // coherent(device)
  id<MTLLibrary> lib = [S->device newLibraryWithSource:[NSString stringWithUTF8String:kSource] options:opts error:&err];
  TORCH_CHECK(lib, "compile failed: ", err ? err.localizedDescription.UTF8String : "?");
  S->tiny = [S->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"tiny"] error:&err];
  S->matvec = [S->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matvec"] error:&err];
  S->touch = [S->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"touch"] error:&err];
  S->side = [S->device newCommandQueue];
  S->pdl = [S->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matvec_pdl"] error:&err];
  S->counters = [S->device newBufferWithLength:4 * 4096 options:MTLResourceStorageModeShared];
  for (int i = 0; i < 96; ++i) S->weights.push_back([S->device newBufferWithLength:1024 * 2304 * 2 options:MTLResourceStorageModePrivate]);
  S->sink = [S->device newBufferWithLength:16 options:MTLResourceStorageModePrivate];
  for (int i = 0; i < 65; ++i) S->act.push_back([S->device newBufferWithLength:65536 * 4 options:MTLResourceStorageModePrivate]);
  S->weight = [S->device newBufferWithLength:1024 * 2304 * 2 options:MTLResourceStorageModePrivate];
}

// mode: 0 serial-chain, 1 conc-chain, 2 serial-indep, 3 conc-indep, 4 encoder-per-op. Returns GPU microseconds.
double run(int64_t mode, int64_t n, int64_t group, int64_t kind) {
  init();
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S->queue commandBuffer];
    const bool concurrent = mode == 1 || mode == 3;
    id<MTLComputeCommandEncoder> enc = nil;
    auto open = [&]() {
      enc = concurrent ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [cb computeCommandEncoder];
    };
    if (mode != 4) open();
    for (int64_t i = 0; i < n; ++i) {
      if (mode == 4) open();
      const bool chain = mode == 0 || mode == 1 || mode == 4;
      // chain: i -> i+1 (true dependency). indep: groups of `group` dispatches reading the same input, distinct outputs
      const int in = chain ? int(i % 64) : int((i / group) % 32) * 2;
      const int out = chain ? int((i + 1) % 64) : int(1 + (i % group) + ((i / group) % 2) * 32) % 65;
      if (kind == 0) {
        [enc setComputePipelineState:S->tiny];
        [enc setBuffer:S->act[in] offset:0 atIndex:0];
        [enc setBuffer:S->act[out] offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(4096, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      } else {
        [enc setComputePipelineState:S->matvec];
        [enc setBuffer:S->weight offset:0 atIndex:0];
        [enc setBuffer:S->act[in] offset:0 atIndex:1];
        [enc setBuffer:S->act[out] offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1024 / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      if (concurrent) {
        const bool barrier = mode == 1 || ((i + 1) % group == 0);  // chain: after every dispatch; indep: between groups
        if (barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      if (mode == 4) [enc endEncoding];
    }
    if (mode != 4) [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e6;
  }
}


// DRAM-bound dependent chain: n matvecs, step i reads weights[i % 96] (96 x 4.7 MB = 453 MB, far past the cache) and the
// previous step's output. With `ahead` > 0, a second command queue runs `touch` kernels over weights[i + ahead] alongside,
// pulling the upcoming weights into the system cache while the chain computes. Returns wall microseconds (both queues).
double chain_prefetch(int64_t n, int64_t ahead, int64_t touch_threads) {
  init();
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (int64_t i = 0; i < n; ++i) {
      [enc setComputePipelineState:S->matvec];
      [enc setBuffer:S->weights[i % 96] offset:0 atIndex:0];
      [enc setBuffer:S->act[i % 64] offset:0 atIndex:1];
      [enc setBuffer:S->act[(i + 1) % 64] offset:0 atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(1024 / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [enc endEncoding];
    id<MTLCommandBuffer> side = nil;
    if (ahead > 0) {
      side = [S->side commandBuffer];
      id<MTLComputeCommandEncoder> e2 = [side computeCommandEncoder];
      for (int64_t i = 0; i < n; ++i) {
        [e2 setComputePipelineState:S->touch];
        [e2 setBuffer:S->weights[(i + ahead) % 96] offset:0 atIndex:0];
        [e2 setBuffer:S->sink offset:0 atIndex:1];
        [e2 dispatchThreads:MTLSizeMake(touch_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      [e2 endEncoding];
    }
    auto t0 = std::chrono::steady_clock::now();
    if (side) [side commit];
    [cb commit];
    [cb waitUntilCompleted];
    auto t1 = std::chrono::steady_clock::now();
    if (side) [side waitUntilCompleted];
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
  }
}


// Dependent chain of n matvecs over distinct weights: mode 0 serial encoder (torch today), mode 1 concurrent encoder
// with no barriers and the device-counter handoff (software PDL). Returns GPU microseconds and a correctness flag
// (the last output vs a serial run is compared from Python).
std::vector<double> chain_pdl(int64_t n, int64_t mode, int64_t tgs) {
  init();
  memset(S->counters.contents, 0, 4 * 4096);
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = mode == 1 ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                                                 : [cb computeCommandEncoder];
    for (int64_t i = 0; i < n; ++i) {
      [enc setComputePipelineState:mode == 1 ? S->pdl : S->matvec];
      [enc setBuffer:S->weights[i % 96] offset:0 atIndex:0];
      [enc setBuffer:S->act[i % 64] offset:0 atIndex:1];
      [enc setBuffer:S->act[(i + 1) % 64] offset:0 atIndex:2];
      if (mode == 1) {
        uint32_t need = i == 0 ? 0 : (uint32_t)tgs;
        [enc setBuffer:S->counters offset:4 * ((i + 4095) % 4096) atIndex:3];
        [enc setBuffer:S->counters offset:4 * (i % 4096) atIndex:4];
        [enc setBytes:&need length:4 atIndex:5];
      }
      [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    uint32_t* c = (uint32_t*)S->counters.contents;
    double complete = mode == 1 ? (c[(n - 1) % 4096] == (uint32_t)tgs ? 1.0 : 0.0) : 1.0;
    return {(cb.GPUEndTime - cb.GPUStartTime) * 1e6, complete};
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("chain_pdl", &chain_pdl); m.def("run", &run); m.def("chain_prefetch", &chain_prefetch); }
