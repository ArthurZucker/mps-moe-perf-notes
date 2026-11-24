// AOTI command-block runner: encodes AOTI Metal kernels into one concurrent
// compute encoder and inserts explicit memoryBarrierWithResources edges.
//
// It overrides the aoti_torch_mps_* shim symbols linked into the AOTI wrapper:
//  - start_encoding sets the MetalKernelFunction's encoder to our concurrent one
//  - set_arg_tensor records the MTLBuffer and whether it is the output (arg 0)
//  - dispatch_* inserts a barrier for read-after-write / write-after-read edges
//  - run_command_block executes the generated lambda
//
// Python calls mps_concurrent_begin()/mps_concurrent_finish() around model.run.
#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/Functions.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace torch::aot_inductor;

namespace {
struct MKF {
  id<MTLComputePipelineState> cps;
  id<MTLFunction> func;
  id<MTLComputeCommandEncoder> encoder;
};

struct Tenant {
  id<MTLCommandQueue> queue = nil;
  id<MTLCommandBuffer> cb = nil;
  id<MTLComputeCommandEncoder> enc = nil;
  uint64_t gen = 0, barrier_gen = 0;
  std::unordered_map<void*, uint64_t> last_write;
  std::unordered_map<void*, uint64_t> last_read;
  std::vector<id<MTLResource>> reads, writes;
  bool in_kernel = false;
};
static Tenant T;
static bool g_conservative = false;

static void ensure() {
  if (T.enc) return;
  id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
  if (!T.queue) T.queue = [device newCommandQueue];
  T.cb = [T.queue commandBuffer];
  T.enc = [T.cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  T.gen = T.barrier_gen = 0;
  T.last_write.clear(); T.last_read.clear();
  T.reads.clear(); T.writes.clear();
}

static void barrier_if_needed() {
  std::vector<id<MTLResource>> need;
  for (auto* b : T.reads) {
    auto it = T.last_write.find((void*)b);
    if (it != T.last_write.end() && it->second > T.barrier_gen) need.push_back(b);
  }
  for (auto* b : T.writes) {
    auto it = T.last_write.find((void*)b);
    if (it != T.last_write.end() && it->second > T.barrier_gen) need.push_back(b);
    auto ir = T.last_read.find((void*)b);
    if (ir != T.last_read.end() && ir->second > T.barrier_gen) need.push_back(b);
  }
  if (g_conservative) need.insert(need.end(), T.reads.begin(), T.reads.end());
  if (!need.empty()) {
    std::sort(need.begin(), need.end());
    need.erase(std::unique(need.begin(), need.end()), need.end());
    [T.enc memoryBarrierWithResources:need.data() count:need.size()];
    T.barrier_gen = T.gen;
  }
}

static void finish_kernel() {
  T.gen++;
  for (auto* b : T.writes) T.last_write[(void*)b] = T.gen;
  for (auto* b : T.reads) T.last_read[(void*)b] = T.gen;
  T.reads.clear(); T.writes.clear(); T.in_kernel = false;
}
} // namespace

extern "C" {

// The AOTI package normally supplies the serialized constant blob. This toy has no
// constants; keep the symbols the wrapper expects, with an empty one-byte blob.
extern const char _binary_constants_bin_start[1] = {0};
extern const char _binary_constants_bin_end[1] = {0};

AOTITorchError aoti_torch_mps_start_encoding(AOTIMetalKernelFunctionHandle handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    ensure();
    auto* f = reinterpret_cast<MKF*>(handle);
    f->encoder = T.enc;
    [T.enc setComputePipelineState:f->cps];
    T.in_kernel = true;
  });
}

AOTITorchError aoti_torch_mps_set_arg_tensor(AOTIMetalKernelFunctionHandle handle, unsigned idx, AtenTensorHandle tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* t = tensor_handle_to_tensor_pointer(tensor);
    TORCH_CHECK(t != nullptr, "Tensor is null.");
    auto* f = reinterpret_cast<MKF*>(handle);
    f->encoder = T.enc;
    auto func = reinterpret_cast<at::native::mps::MetalKernelFunction*>(handle);
    func->setArg(idx, *t);
    id<MTLBuffer> b = at::native::mps::getMTLBufferStorage(*t);
    if (b) (idx == 0 ? T.writes : T.reads).push_back(b);
  });
}

AOTITorchError aoti_torch_mps_set_arg_int(AOTIMetalKernelFunctionHandle handle, unsigned idx, int64_t val) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto func = reinterpret_cast<at::native::mps::MetalKernelFunction*>(handle);
    func->setArg(idx, val);
  });
}

AOTITorchError aoti_torch_mps_dispatch_single(AOTIMetalKernelFunctionHandle handle, uint64_t length) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    ensure();
    barrier_if_needed();
    auto func = reinterpret_cast<at::native::mps::MetalKernelFunction*>(handle);
    func->dispatch(length);
    finish_kernel();
  });
}

AOTITorchError aoti_torch_mps_dispatch_single_with_group_size(AOTIMetalKernelFunctionHandle handle, uint64_t length, uint64_t group_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    ensure();
    barrier_if_needed();
    auto func = reinterpret_cast<at::native::mps::MetalKernelFunction*>(handle);
    func->dispatch(length, group_size);
    finish_kernel();
  });
}

AOTITorchError aoti_torch_mps_dispatch_array(AOTIMetalKernelFunctionHandle handle, const uint64_t* length, size_t length_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    ensure();
    barrier_if_needed();
    auto func = reinterpret_cast<at::native::mps::MetalKernelFunction*>(handle);
    c10::ArrayRef<uint64_t> l(length, length_size);
    func->dispatch(l);
    finish_kernel();
  });
}

AOTITorchError aoti_torch_mps_dispatch_array_with_group_size(AOTIMetalKernelFunctionHandle handle, const uint64_t* length, size_t length_size, const uint64_t* group_size, size_t group_size_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    ensure();
    barrier_if_needed();
    auto func = reinterpret_cast<at::native::mps::MetalKernelFunction*>(handle);
    c10::ArrayRef<uint64_t> l(length, length_size);
    c10::ArrayRef<uint64_t> g(group_size, group_size_size);
    func->dispatch(l, g);
    finish_kernel();
  });
}

void aoti_torch_mps_shared_callback(AOTIMetalKernelFunctionHandle func, void* user_data) {
  auto* f = static_cast<std::function<void(AOTIMetalKernelFunctionHandle)>*>(user_data);
  (*f)(func);
}

AOTITorchError aoti_torch_mps_run_command_block(AOTIMetalKernelFunctionHandle handle,
                                                aoti_torch_mps_command_block_callback_t callback,
                                                void* user_data) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    ensure();
    T.reads.clear(); T.writes.clear(); T.in_kernel = true;
    callback(handle, user_data);
    if (T.in_kernel) finish_kernel();
  });
}

void mps_concurrent_finish();

void mps_concurrent_begin() {
  if (T.enc) mps_concurrent_finish();
  // AOTI can mix our Metal command buffer with PyTorch-stream ops (e.g. cumsum).
  // Drain the PyTorch stream before opening a new command buffer so a mixed op
  // that produced our inputs has completed.
  if (auto* s = at::mps::getCurrentMPSStream()) s->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
  ensure();
}

void mps_concurrent_set_conservative(int on) { g_conservative = on != 0; }

void mps_concurrent_finish() {
  if (!T.enc) return;
  [T.enc endEncoding];
  [T.cb commit];
  [T.cb waitUntilCompleted];
  T.enc = nil; T.cb = nil;
  T.last_write.clear(); T.last_read.clear();
  T.reads.clear(); T.writes.clear();
  T.gen = T.barrier_gen = 0;
}


// ---------------------------------------------------------------------------
// Metal cumsum: the AOTI cumsum op otherwise runs on PyTorch's stream and forces
// a flush around every mixed op. One 1-D float32 inclusive scan in three stages,
// all in our concurrent encoder.
// ---------------------------------------------------------------------------
static const char* kCumsumSrc = R"MTL(
#include <metal_stdlib>
using namespace metal;
kernel void cumsum_local(device const float* x [[buffer(0)]],
                         device float* y [[buffer(1)]],
                         device float* blocks [[buffer(2)]],
                         constant uint& n [[buffer(3)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tgs [[threads_per_threadgroup]]) {
  threadgroup float scan[1024];
  uint base = tg * tgs;
  uint i = base + tid;
  scan[tid] = i < n ? x[i] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint off = 1; off < tgs; off <<= 1) {
    float prev = tid >= off ? scan[tid - off] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scan[tid] += prev;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (i < n) y[i] = scan[tid];
  if (tid == tgs - 1) blocks[tg] = scan[tid];
}
kernel void cumsum_blocks(device float* blocks [[buffer(0)]],
                          constant uint& nb [[buffer(1)]],
                          uint tid [[thread_position_in_threadgroup]]) {
  if (tid == 0) { float s = 0.0f; for (uint i = 0; i < nb; ++i) { s += blocks[i]; blocks[i] = s; } }
}
kernel void cumsum_add(device float* y [[buffer(0)]],
                       device const float* blocks [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint tgs [[threads_per_threadgroup]]) {
  if (tg == 0) return;
  uint i = tg * tgs + tid;
  if (i < n) y[i] += blocks[tg - 1];
}
kernel void cumsum_seq(device const float* x [[buffer(0)]],
                       device float* y [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       uint tid [[thread_position_in_threadgroup]]) {
  if (tid != 0) return;
  float s = 0.0f;
  for (uint i = 0; i < n; ++i) { s += x[i]; y[i] = s; }
}
)MTL";

static at::native::mps::DynamicMetalShaderLibrary* g_cumsum_lib = nullptr;
static at::native::mps::MetalKernelFunction* g_cum_local = nullptr;
static at::native::mps::MetalKernelFunction* g_cum_blocks = nullptr;
static at::native::mps::MetalKernelFunction* g_cum_add = nullptr;
static at::native::mps::MetalKernelFunction* g_cum_seq = nullptr;

static void ensure_cumsum() {
  if (g_cum_local) return;
  g_cumsum_lib = new at::native::mps::DynamicMetalShaderLibrary(kCumsumSrc);
  g_cum_local = g_cumsum_lib->getCachedKernelFunctionPtr("cumsum_local");
  g_cum_blocks = g_cumsum_lib->getCachedKernelFunctionPtr("cumsum_blocks");
  g_cum_add = g_cumsum_lib->getCachedKernelFunctionPtr("cumsum_add");
  g_cum_seq = g_cumsum_lib->getCachedKernelFunctionPtr("cumsum_seq");
}

static void use_encoder(at::native::mps::MetalKernelFunction* f) {
  auto* m = reinterpret_cast<MKF*>(f);
  m->encoder = T.enc;
  [T.enc setComputePipelineState:m->cps];
}

AOTITorchError aoti_torch_mps_cumsum(AtenTensorHandle self, int64_t dim, int32_t* dtype, AtenTensorHandle* ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* tp = tensor_handle_to_tensor_pointer(self);
    TORCH_CHECK(tp != nullptr, "cumsum input is null");
    const at::Tensor& in = *tp;
    TORCH_CHECK(in.dim() == 1 && dim == 0 && in.scalar_type() == at::kFloat && in.is_contiguous(),
                "metal cumsum expects a contiguous 1-D float32 tensor");
    ensure();
    ensure_cumsum();
    at::Tensor out = at::empty_like(in);
    const uint32_t n = (uint32_t)in.numel();
    const uint32_t tgs = 256;
    const uint32_t nb = (n + tgs - 1) / tgs;
    at::Tensor blocks = at::empty({(int64_t)nb}, in.options());
    // The producer (sin, etc.) was encoded earlier in this concurrent encoder.
    [T.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    id<MTLBuffer> xb = at::native::mps::getMTLBufferStorage(in);
    id<MTLBuffer> yb = at::native::mps::getMTLBufferStorage(out);
    id<MTLBuffer> bb = at::native::mps::getMTLBufferStorage(blocks);
    if (n <= 1024) {
      // exact left-to-right order, matching the serial reference
      use_encoder(g_cum_seq);
      g_cum_seq->setArg(0, in);
      g_cum_seq->setArg(1, out);
      g_cum_seq->setArg(2, n);
      g_cum_seq->dispatch(1, 1);
    } else {
      use_encoder(g_cum_local);
      g_cum_local->setArg(0, in);
      g_cum_local->setArg(1, out);
      g_cum_local->setArg(2, blocks);
      g_cum_local->setArg(3, n);
      g_cum_local->dispatch(nb * tgs, tgs);
      [T.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      use_encoder(g_cum_blocks);
      g_cum_blocks->setArg(0, blocks);
      g_cum_blocks->setArg(1, nb);
      g_cum_blocks->dispatch(tgs, tgs);
      [T.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      use_encoder(g_cum_add);
      g_cum_add->setArg(0, out);
      g_cum_add->setArg(1, blocks);
      g_cum_add->setArg(2, n);
      g_cum_add->dispatch(nb * tgs, tgs);
    }
    [T.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    T.barrier_gen = T.gen;
    T.last_write[(void*)yb] = T.gen;
    T.last_read[(void*)xb] = T.gen;
    *ret0 = new_tensor_handle(std::move(out));
  });
}

} // extern "C"
