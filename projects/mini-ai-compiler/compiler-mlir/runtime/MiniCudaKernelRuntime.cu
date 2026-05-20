#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <cstdint>
#include <cstdio>

#ifdef MINI_ENABLE_NVTX
#include "nvtx3/nvToolsExt.h"
#endif

#ifdef _WIN32
#define MINI_CUDA_WRAPPERS_EXPORT __declspec(dllexport)
#else
#define MINI_CUDA_WRAPPERS_EXPORT __attribute__((visibility("default")))
#endif

extern "C" void miniPerfAppendKernelTimingMs(double timingMs);

namespace {

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

static void reportCuda(cudaError_t status, const char *expr) {
  if (status != cudaSuccess)
    fprintf(stderr, "'%s' failed with '%s'\n", expr, cudaGetErrorString(status));
}

static void reportCublas(cublasStatus_t status, const char *expr) {
  if (status != CUBLAS_STATUS_SUCCESS)
    fprintf(stderr, "'%s' failed with cuBLAS status %d\n", expr,
            static_cast<int>(status));
}

#define CUDA_REPORT_IF_ERROR(expr) reportCuda((expr), #expr)
#define CUBLAS_REPORT_IF_ERROR(expr) reportCublas((expr), #expr)

__global__ void miniLinearReluKernel(const float *input, const float *weight,
                                     const float *bias, float *output, int m,
                                     int n, int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n)
    return;

  float acc = bias[col];
  for (int kk = 0; kk < k; ++kk)
    acc += input[row * k + kk] * weight[col * k + kk];
  output[row * n + col] = fmaxf(acc, 0.0f);
}

__global__ void miniBiasReluKernel(float *output, const float *bias, int m,
                                   int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = m * n;
  if (index >= total)
    return;
  int col = index % n;
  output[index] = fmaxf(output[index] + bias[col], 0.0f);
}

template <typename LaunchFn>
static void timeKernelSequence(cudaStream_t stream, LaunchFn launch) {
  cudaEvent_t startEvent = nullptr;
  cudaEvent_t stopEvent = nullptr;
  CUDA_REPORT_IF_ERROR(cudaEventCreate(&startEvent));
  CUDA_REPORT_IF_ERROR(cudaEventCreate(&stopEvent));
  CUDA_REPORT_IF_ERROR(cudaEventRecord(startEvent, stream));
  launch();
  CUDA_REPORT_IF_ERROR(cudaGetLastError());
  CUDA_REPORT_IF_ERROR(cudaEventRecord(stopEvent, stream));
  CUDA_REPORT_IF_ERROR(cudaEventSynchronize(stopEvent));
  float elapsedMs = 0.0f;
  CUDA_REPORT_IF_ERROR(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
  miniPerfAppendKernelTimingMs(static_cast<double>(elapsedMs));
  CUDA_REPORT_IF_ERROR(cudaEventDestroy(startEvent));
  CUDA_REPORT_IF_ERROR(cudaEventDestroy(stopEvent));
}

static cudaStream_t asCudaStream(void *stream) {
  return reinterpret_cast<cudaStream_t>(stream);
}

} // namespace

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mini_cuda_linear_relu_f32(void *input, void *weight, void *bias, void *output,
                          int64_t m, int64_t n, int64_t k, void *stream) {
  NvtxRange range("op/linear_relu/cuda_hand");
  cudaStream_t cudaStream = asCudaStream(stream);
  dim3 block(16, 16);
  dim3 grid((static_cast<unsigned>(n) + block.x - 1) / block.x,
            (static_cast<unsigned>(m) + block.y - 1) / block.y);

  timeKernelSequence(cudaStream, [&] {
    miniLinearReluKernel<<<grid, block, 0, cudaStream>>>(
        static_cast<const float *>(input), static_cast<const float *>(weight),
        static_cast<const float *>(bias), static_cast<float *>(output),
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
  });
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mini_cublas_linear_relu_f32(void *input, void *weight, void *bias, void *output,
                            int64_t m, int64_t n, int64_t k, void *stream) {
  NvtxRange range("op/linear_relu/cublas");
  cudaStream_t cudaStream = asCudaStream(stream);
  cublasHandle_t handle = nullptr;
  CUBLAS_REPORT_IF_ERROR(cublasCreate(&handle));
  CUBLAS_REPORT_IF_ERROR(cublasSetStream(handle, cudaStream));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  int mi = static_cast<int>(m);
  int ni = static_cast<int>(n);
  int ki = static_cast<int>(k);
  int total = mi * ni;
  int block = 256;
  int grid = (total + block - 1) / block;

  timeKernelSequence(cudaStream, [&] {
    // Row-major C[M,N] is column-major C^T[N,M]. With row-major A[M,K]
    // and W[N,K], use column-major views A_col[K,M] and W_col[K,N].
    CUBLAS_REPORT_IF_ERROR(cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, ni, mi, ki, &alpha,
        static_cast<const float *>(weight), ki, static_cast<const float *>(input),
        ki, &beta, static_cast<float *>(output), ni));
    miniBiasReluKernel<<<grid, block, 0, cudaStream>>>(
        static_cast<float *>(output), static_cast<const float *>(bias), mi, ni);
  });

  CUBLAS_REPORT_IF_ERROR(cublasDestroy(handle));
}

