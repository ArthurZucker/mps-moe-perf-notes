// GPU-side waits on host-signalled Metal shared events inside PyTorch's MPS stream, and CPU pointers into shared MPS
// tensors: lets the host react to a layer's routing (read from shared memory) without draining the GPU queue.
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#import <Metal/Metal.h>
#include <chrono>
#include <vector>
#include <mutex>
#include <tuple>
#import <QuartzCore/QuartzCore.h>
#include <pybind11/stl.h>

static std::vector<id<MTLSharedEvent>> events;

int64_t event_new() {
  id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
  events.push_back([device newSharedEvent]);
  return (int64_t)events.size() - 1;
}

// Queue a wait: work submitted to the stream after this runs once the event reaches `value`. Submits the buffer so the
// device runs the work before the wait without waiting for a host sync.
void gpu_wait(int64_t handle, int64_t value, bool commit) {
  pybind11::gil_scoped_release release;  // a commit can block on the in-flight buffer limit: let the host thread run
  auto* stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^() {
    stream->endKernelCoalescing();
    [stream->commandBuffer() encodeWaitForEvent:events[handle] value:(uint64_t)value];
  });
  if (commit) stream->synchronize(at::mps::SyncType::COMMIT);
}

// End the open compute encoder and submit: work queued so far (e.g. a routing copy) runs and its writes become visible.
void flush() {
  pybind11::gil_scoped_release release;
  auto* stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^() { stream->endKernelCoalescing(); });
  stream->synchronize(at::mps::SyncType::COMMIT);
}

void cpu_signal(int64_t handle, int64_t value) { events[handle].signaledValue = (uint64_t)value; }

int64_t signaled(int64_t handle) { return (int64_t)events[handle].signaledValue; }

// Address of the tensor's first element in CPU-visible memory (shared storage only), else 0.
int64_t cpu_address(const at::Tensor& t) {
  id<MTLBuffer> buffer = at::native::mps::getMTLBufferStorage(t);
  if (buffer.storageMode != MTLStorageModeShared) return 0;
  return (int64_t)((char*)buffer.contents + t.storage_offset() * t.element_size());
}

// Spin (GIL released) until the int32 at `address` equals `value`; returns false after `timeout_us`.
bool spin_until(int64_t address, int32_t value, int64_t timeout_us) {
  pybind11::gil_scoped_release release;
  volatile int32_t* p = (volatile int32_t*)address;
  auto start = std::chrono::steady_clock::now();
  while (*p != value) {
    if (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count() > timeout_us)
      return false;
  }
  return true;
}

// Spin (GIL released) until none of the `n` int32 at `address` is -1 (a whole routing landed); false on timeout.
bool spin_until_landed(int64_t address, int64_t n, int64_t timeout_us) {
  pybind11::gil_scoped_release release;
  volatile int32_t* p = (volatile int32_t*)address;
  auto start = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < n;) {
    if (p[i] != -1) { ++i; continue; }
    if (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count() > timeout_us)
      return false;
  }
  return true;
}

// The hit path of a streamed layer, GIL released: wait until its [routing (n) | slots pass 1 read (n) | guess (g)]
// block landed (int32, -1 = not landed). If no slot is `sentinel` (every expert pass 1 needed was resident): stamp the
// slots' last use and signal the event (returns 1); the caller still reads the block, then resets it. A miss: 0.
// Timeout: -1.
int64_t resolve_hits(int64_t block, int64_t n, int64_t g, int64_t sentinel, int64_t stamps, int64_t stamp,
                     int64_t handle, int64_t counter, int64_t timeout_us) {
  pybind11::gil_scoped_release release;
  volatile int32_t* b = (volatile int32_t*)block;
  int64_t* st = (int64_t*)stamps;
  auto start = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < 2 * n + g;) {
    if (b[i] != -1) { ++i; continue; }
    if (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count() > timeout_us)
      return -1;
  }
  for (int64_t i = n; i < 2 * n; ++i)
    if (b[i] == sentinel) return 0;
  for (int64_t i = n; i < 2 * n; ++i) st[b[i]] = stamp;
  events[handle].signaledValue = (uint64_t)counter;
  return 1;
}

int64_t storage_mode(const at::Tensor& t) { return (int64_t)at::native::mps::getMTLBufferStorage(t).storageMode; }

// GPU timestamps of committed command buffers (low overhead, unlike Metal System Trace): flush_timed() ends the open
// encoder, attaches a completion handler that records (GPUStartTime, GPUEndTime, host commit time) and commits.
static std::mutex timing_mutex;
static std::vector<std::tuple<double, double, double, int64_t>> timings_v;

void flush_timed(int64_t tag) {
  pybind11::gil_scoped_release release;
  auto* stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^() {
    stream->endKernelCoalescing();
    double host = CACurrentMediaTime();
    [stream->commandBuffer() addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      std::lock_guard<std::mutex> lock(timing_mutex);
      timings_v.emplace_back(cb.GPUStartTime, cb.GPUEndTime, host, tag);
    }];
  });
  stream->synchronize(at::mps::SyncType::COMMIT);
}

std::vector<std::tuple<double, double, double, int64_t>> timings() {
  std::lock_guard<std::mutex> lock(timing_mutex);
  auto out = timings_v;
  timings_v.clear();
  return out;
}

// Passive tracer: swizzle -commit on the device's concrete command-buffer class so every command buffer the process commits
// (torch's stream, MPSGraph, ours) records (GPUStartTime, GPUEndTime, commit host time, tag = -2) on completion.
#import <objc/runtime.h>
static IMP original_commit = nullptr;
static bool tracer_on = false;
static void traced_commit(id self, SEL _cmd) {
  if (tracer_on) {
    double host = CACurrentMediaTime();
    [(id<MTLCommandBuffer>)self addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      std::lock_guard<std::mutex> lock(timing_mutex);
      timings_v.emplace_back(cb.GPUStartTime, cb.GPUEndTime, host, -2);
    }];
  }
  ((void (*)(id, SEL))original_commit)(self, _cmd);
}
std::string trace_all(bool on) {
  if (!original_commit) {
    id<MTLCommandQueue> q = [at::mps::MPSDevice::getInstance()->device() newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    Class cls = [cb class];
    Method m = class_getInstanceMethod(cls, @selector(commit));
    original_commit = method_setImplementation(m, (IMP)traced_commit);
    [cb commit];
    tracer_on = on;
    return std::string(class_getName(cls));
  }
  tracer_on = on;
  return "";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("trace_all", &trace_all);
  m.def("flush_timed", &flush_timed);
  m.def("timings", &timings);
  m.def("event_new", &event_new);
  m.def("gpu_wait", &gpu_wait);
  m.def("cpu_signal", &cpu_signal);
  m.def("flush", &flush);
  m.def("signaled", &signaled);
  m.def("cpu_address", &cpu_address);
  m.def("storage_mode", &storage_mode);
  m.def("spin_until", &spin_until);
  m.def("spin_until_landed", &spin_until_landed);
  m.def("resolve_hits", &resolve_hits);
}
