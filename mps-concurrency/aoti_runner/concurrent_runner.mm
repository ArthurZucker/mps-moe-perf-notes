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

} // extern "C"
