#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda.h"
#ifdef MINI_ENABLE_NVTX
#include "nvtx3/nvToolsExt.h"
#endif

#ifdef _WIN32
#define MINI_CUDA_WRAPPERS_EXPORT __declspec(dllexport)
#else
#define MINI_CUDA_WRAPPERS_EXPORT __attribute__((visibility("default")))
#endif

class NvtxRange {
public:
  explicit NvtxRange(const char *name) {
#ifdef MINI_ENABLE_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }
  ~NvtxRange() {
#ifdef MINI_ENABLE_NVTX
    nvtxRangePop();
#endif
  }
};

#define CUDA_REPORT_IF_ERROR(expr)                                             \
  [](CUresult result) {                                                        \
    if (result == CUDA_SUCCESS)                                                \
      return;                                                                  \
    const char *name = nullptr;                                                \
    cuGetErrorName(result, &name);                                             \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
  }(expr)

thread_local static int32_t defaultDevice = 0;
thread_local static std::vector<double> kernelTimingsMs;

static CUdevice getDefaultCuDevice() {
  CUdevice device = 0;
  CUDA_REPORT_IF_ERROR(cuDeviceGet(&device, defaultDevice));
  return device;
}

class ScopedContext {
public:
  ScopedContext() {
    static CUcontext context = [] {
      CUDA_REPORT_IF_ERROR(cuInit(0));
      CUcontext ctx = nullptr;
      CUDA_REPORT_IF_ERROR(cuDevicePrimaryCtxRetain(&ctx, getDefaultCuDevice()));
      return ctx;
    }();
    CUDA_REPORT_IF_ERROR(cuCtxPushCurrent(context));
  }

  ~ScopedContext() { CUDA_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr)); }
};

extern "C" MINI_CUDA_WRAPPERS_EXPORT void mgpuSetDefaultDevice(int32_t device) {
  defaultDevice = device;
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void miniPerfResetKernelTimings() {
  kernelTimingsMs.clear();
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT int64_t miniPerfGetKernelTimingCount() {
  return static_cast<int64_t>(kernelTimingsMs.size());
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT double
miniPerfGetKernelTimingMs(int64_t index) {
  if (index < 0 || static_cast<size_t>(index) >= kernelTimingsMs.size())
    return 0.0;
  return kernelTimingsMs[static_cast<size_t>(index)];
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
miniPerfAppendKernelTimingMs(double timingMs) {
  kernelTimingsMs.push_back(timingMs);
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT CUmodule mgpuModuleLoad(void *data,
                                                             size_t) {
  NvtxRange range("module_load");
  ScopedContext scopedContext;
  CUmodule module = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
  return module;
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT CUmodule mgpuModuleLoadJIT(void *data,
                                                                int optLevel) {
  NvtxRange range("module_load_jit");
  ScopedContext scopedContext;
  CUmodule module = nullptr;
  char jitErrorBuffer[4096] = {0};
  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                               CU_JIT_OPTIMIZATION_LEVEL};
  void *jitOptionVals[] = {jitErrorBuffer,
                           reinterpret_cast<void *>(sizeof(jitErrorBuffer)),
                           reinterpret_cast<void *>(static_cast<intptr_t>(optLevel))};
  CUresult result = cuModuleLoadDataEx(&module, data, 3, jitOptions, jitOptionVals);
  if (result != CUDA_SUCCESS)
    fprintf(stderr, "JIT compilation failed with: '%s'\n", jitErrorBuffer);
  CUDA_REPORT_IF_ERROR(result);
  return module;
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void mgpuModuleUnload(CUmodule module) {
  CUDA_REPORT_IF_ERROR(cuModuleUnload(module));
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT CUfunction
mgpuModuleGetFunction(CUmodule module, const char *name) {
  CUfunction function = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleGetFunction(&function, module, name));
  return function;
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT CUstream mgpuStreamCreate() {
  ScopedContext scopedContext;
  CUstream stream = nullptr;
  CUDA_REPORT_IF_ERROR(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  return stream;
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void mgpuStreamDestroy(CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuStreamDestroy(stream));
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void mgpuStreamSynchronize(CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mgpuStreamWaitEvent(CUstream stream, CUevent event) {
  CUDA_REPORT_IF_ERROR(cuStreamWaitEvent(stream, event, 0));
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT CUevent mgpuEventCreate() {
  ScopedContext scopedContext;
  CUevent event = nullptr;
  CUDA_REPORT_IF_ERROR(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
  return event;
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void mgpuEventDestroy(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventDestroy(event));
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void mgpuEventSynchronize(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventSynchronize(event));
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mgpuEventRecord(CUevent event, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuEventRecord(event, stream));
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void *
mgpuMemAlloc(uint64_t sizeBytes, CUstream, bool isHostShared) {
  ScopedContext scopedContext;
  CUdeviceptr ptr = 0;
  if (sizeBytes == 0)
    return reinterpret_cast<void *>(ptr);
  if (isHostShared) {
    CUDA_REPORT_IF_ERROR(cuMemAllocManaged(&ptr, sizeBytes, CU_MEM_ATTACH_GLOBAL));
    return reinterpret_cast<void *>(ptr);
  }
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&ptr, sizeBytes));
  return reinterpret_cast<void *>(ptr);
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void mgpuMemFree(void *ptr, CUstream) {
  CUDA_REPORT_IF_ERROR(cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)));
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mgpuMemcpy(void *dst, void *src, size_t sizeBytes, CUstream stream) {
  NvtxRange range("memcpy");
  CUDA_REPORT_IF_ERROR(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst),
                                     reinterpret_cast<CUdeviceptr>(src), sizeBytes,
                                     stream));
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mgpuMemset32(void *dst, unsigned int value, size_t count, CUstream stream) {
  CUDA_REPORT_IF_ERROR(
      cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(dst), value, count, stream));
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mgpuMemset16(void *dst, unsigned short value, size_t count, CUstream stream) {
  CUDA_REPORT_IF_ERROR(
      cuMemsetD16Async(reinterpret_cast<CUdeviceptr>(dst), value, count, stream));
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mgpuLaunchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                 intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                 intptr_t blockZ, int32_t smem, CUstream stream, void **params,
                 void **extra, size_t) {
  NvtxRange range("kernel_launch");
  ScopedContext scopedContext;
  CUevent startEvent = nullptr;
  CUevent stopEvent = nullptr;
  CUDA_REPORT_IF_ERROR(cuEventCreate(&startEvent, CU_EVENT_DEFAULT));
  CUDA_REPORT_IF_ERROR(cuEventCreate(&stopEvent, CU_EVENT_DEFAULT));
  if (smem > 0) {
    int32_t maxShmem = 0;
    CUdevice device = getDefaultCuDevice();
    CUDA_REPORT_IF_ERROR(cuDeviceGetAttribute(
        &maxShmem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device));
    if (maxShmem < smem) {
      fprintf(stderr,
              "Requested shared memory (%d bytes) exceeds device limit (%d bytes)\n",
              smem, maxShmem);
    }
    CUDA_REPORT_IF_ERROR(cuFuncSetAttribute(
      function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
  }
  CUDA_REPORT_IF_ERROR(cuEventRecord(startEvent, stream));
  CUDA_REPORT_IF_ERROR(cuLaunchKernel(function, gridX, gridY, gridZ, blockX,
                                      blockY, blockZ, smem, stream, params,
                                      extra));
  CUDA_REPORT_IF_ERROR(cuEventRecord(stopEvent, stream));
  CUDA_REPORT_IF_ERROR(cuEventSynchronize(stopEvent));
  float elapsedMs = 0.0f;
  CUDA_REPORT_IF_ERROR(cuEventElapsedTime(&elapsedMs, startEvent, stopEvent));
  kernelTimingsMs.push_back(static_cast<double>(elapsedMs));
  CUDA_REPORT_IF_ERROR(cuEventDestroy(startEvent));
  CUDA_REPORT_IF_ERROR(cuEventDestroy(stopEvent));
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuMemHostRegister(ptr, sizeBytes, 0));
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mgpuMemHostUnregister(void *ptr) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuMemHostUnregister(ptr));
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mgpuMemHostRegisterMemRef(int64_t rank, StridedMemRefType<char, 1> *descriptor,
                          int64_t elementSizeBytes) {
#ifdef _WIN32
  int64_t *denseStrides =
      static_cast<int64_t *>(_alloca(rank * static_cast<int64_t>(sizeof(int64_t))));
#else
  int64_t *denseStrides =
      static_cast<int64_t *>(alloca(rank * static_cast<int64_t>(sizeof(int64_t))));
#endif
  int64_t *sizes = descriptor->sizes;
  for (int64_t i = rank - 1, runningStride = 1; i >= 0; --i) {
    denseStrides[i] = runningStride;
    runningStride *= sizes[i];
  }
  uint64_t sizeBytes = sizes[0] * denseStrides[0] * elementSizeBytes;
  int64_t *strides = &sizes[rank];
  for (int64_t i = 0; i < rank; ++i)
    assert(strides[i] == denseStrides[i] && "expected densely packed memref");

  auto *ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostRegister(ptr, sizeBytes);
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mgpuMemHostUnregisterMemRef(int64_t, StridedMemRefType<char, 1> *descriptor,
                            int64_t elementSizeBytes) {
  auto *ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostUnregister(ptr);
}
