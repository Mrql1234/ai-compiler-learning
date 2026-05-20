#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

#ifdef MINI_ENABLE_NVTX
#include "nvtx3/nvToolsExt.h"
#endif

#ifdef _WIN32
#define MINI_CUDA_WRAPPERS_EXPORT __declspec(dllexport)
#else
#define MINI_CUDA_WRAPPERS_EXPORT __attribute__((visibility("default")))
#endif

extern "C" void miniPerfAppendKernelTimingMs(double timingMs);
extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mini_cuda_linear_relu_f32(void *input, void *weight, void *bias, void *output,
                          int64_t m, int64_t n, int64_t k, void *stream);
extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mini_cublas_linear_relu_f32(void *input, void *weight, void *bias, void *output,
                            int64_t m, int64_t n, int64_t k, void *stream);

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

static float deterministicInput(int64_t row, int64_t col) {
  return static_cast<float>(((row * 17 + col * 13) % 23) - 11) / 23.0f;
}

static float deterministicWeight(int64_t row, int64_t col) {
  return static_cast<float>(((row * 19 + col * 7) % 29) - 14) / 29.0f;
}

static float deterministicBias(int64_t col) {
  return 0.25f + static_cast<float>((col % 7) - 3) / 31.0f;
}

struct LinearReluProblem {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  cudaStream_t stream = nullptr;
  float *input = nullptr;
  float *weight = nullptr;
  float *bias = nullptr;
  float *output = nullptr;
};

static void fillLinearReluInputs(const char *dataProfile, int64_t m, int64_t n,
                                 int64_t k, float *input, float *weight,
                                 float *bias) {
  const char *profile = dataProfile ? dataProfile : "deterministic";
  std::memset(input, 0, static_cast<size_t>(m * k) * sizeof(float));
  std::memset(weight, 0, static_cast<size_t>(n * k) * sizeof(float));
  std::memset(bias, 0, static_cast<size_t>(n) * sizeof(float));

  if (std::strcmp(profile, "gpu_runner_demo") == 0 && m == 2 && n == 8 &&
      k == 4) {
    input[0] = 1.0f;
    input[1] = 1.0f;
    weight[0] = 1.0f;
    weight[1] = 2.0f;
    bias[0] = 0.5f;
    return;
  }

  for (int64_t row = 0; row < m; ++row)
    for (int64_t col = 0; col < k; ++col)
      input[row * k + col] = deterministicInput(row, col);
  for (int64_t row = 0; row < n; ++row)
    for (int64_t col = 0; col < k; ++col)
      weight[row * k + col] = deterministicWeight(row, col);
  for (int64_t col = 0; col < n; ++col)
    bias[col] = deterministicBias(col);
}

} // namespace

extern "C" MINI_CUDA_WRAPPERS_EXPORT void *
miniCreateLinearReluF32Problem(int64_t m, int64_t n, int64_t k,
                               const char *dataProfile) {
  NvtxRange range("prepare_inputs");
  CUDA_REPORT_IF_ERROR(cudaSetDevice(0));

  auto *problem = new LinearReluProblem;
  problem->m = m;
  problem->n = n;
  problem->k = k;
  CUDA_REPORT_IF_ERROR(
      cudaStreamCreateWithFlags(&problem->stream, cudaStreamNonBlocking));

  const size_t inputElements = static_cast<size_t>(m * k);
  const size_t weightElements = static_cast<size_t>(n * k);
  const size_t biasElements = static_cast<size_t>(n);
  const size_t outputElements = static_cast<size_t>(m * n);
  float *hostInput = new float[inputElements];
  float *hostWeight = new float[weightElements];
  float *hostBias = new float[biasElements];
  fillLinearReluInputs(dataProfile, m, n, k, hostInput, hostWeight, hostBias);

  CUDA_REPORT_IF_ERROR(
      cudaMalloc(&problem->input, inputElements * sizeof(float)));
  CUDA_REPORT_IF_ERROR(
      cudaMalloc(&problem->weight, weightElements * sizeof(float)));
  CUDA_REPORT_IF_ERROR(cudaMalloc(&problem->bias, biasElements * sizeof(float)));
  CUDA_REPORT_IF_ERROR(
      cudaMalloc(&problem->output, outputElements * sizeof(float)));
  CUDA_REPORT_IF_ERROR(cudaMemcpyAsync(problem->input, hostInput,
                                       inputElements * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       problem->stream));
  CUDA_REPORT_IF_ERROR(cudaMemcpyAsync(problem->weight, hostWeight,
                                       weightElements * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       problem->stream));
  CUDA_REPORT_IF_ERROR(cudaMemcpyAsync(problem->bias, hostBias,
                                       biasElements * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       problem->stream));
  CUDA_REPORT_IF_ERROR(cudaStreamSynchronize(problem->stream));

  delete[] hostInput;
  delete[] hostWeight;
  delete[] hostBias;
  return problem;
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
miniRunLinearReluF32Problem(void *opaqueProblem, const char *backend) {
  auto *problem = static_cast<LinearReluProblem *>(opaqueProblem);
  if (!problem)
    return;
  const char *selectedBackend = backend ? backend : "cuda_hand";
  if (std::strcmp(selectedBackend, "cublas") == 0) {
    mini_cublas_linear_relu_f32(problem->input, problem->weight, problem->bias,
                                problem->output, problem->m, problem->n,
                                problem->k, problem->stream);
    return;
  }
  mini_cuda_linear_relu_f32(problem->input, problem->weight, problem->bias,
                            problem->output, problem->m, problem->n,
                            problem->k, problem->stream);
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT float
miniCopyFirstLinearReluF32Result(void *opaqueProblem) {
  auto *problem = static_cast<LinearReluProblem *>(opaqueProblem);
  if (!problem)
    return 0.0f;
  NvtxRange range("memcpy_d2h");
  float result = 0.0f;
  CUDA_REPORT_IF_ERROR(cudaMemcpyAsync(&result, problem->output, sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       problem->stream));
  CUDA_REPORT_IF_ERROR(cudaStreamSynchronize(problem->stream));
  return result;
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
miniDestroyLinearReluF32Problem(void *opaqueProblem) {
  auto *problem = static_cast<LinearReluProblem *>(opaqueProblem);
  if (!problem)
    return;
  cudaFree(problem->input);
  cudaFree(problem->weight);
  cudaFree(problem->bias);
  cudaFree(problem->output);
  cudaStreamDestroy(problem->stream);
  delete problem;
}

static void runLinearReluMemRef(const char *backend, float *input,
                                int64_t inputOffset, float *weight,
                                int64_t weightOffset, float *bias,
                                int64_t biasOffset, float *output,
                                int64_t outputOffset, int64_t m, int64_t n,
                                int64_t k) {
  NvtxRange range("backend/runtime_call_memref");
  CUDA_REPORT_IF_ERROR(cudaSetDevice(0));
  cudaStream_t stream = nullptr;
  CUDA_REPORT_IF_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const size_t inputBytes = static_cast<size_t>(m * k) * sizeof(float);
  const size_t weightBytes = static_cast<size_t>(n * k) * sizeof(float);
  const size_t biasBytes = static_cast<size_t>(n) * sizeof(float);
  const size_t outputBytes = static_cast<size_t>(m * n) * sizeof(float);
  float *deviceInput = nullptr;
  float *deviceWeight = nullptr;
  float *deviceBias = nullptr;
  float *deviceOutput = nullptr;
  CUDA_REPORT_IF_ERROR(cudaMalloc(&deviceInput, inputBytes));
  CUDA_REPORT_IF_ERROR(cudaMalloc(&deviceWeight, weightBytes));
  CUDA_REPORT_IF_ERROR(cudaMalloc(&deviceBias, biasBytes));
  CUDA_REPORT_IF_ERROR(cudaMalloc(&deviceOutput, outputBytes));

  CUDA_REPORT_IF_ERROR(cudaMemcpyAsync(deviceInput, input + inputOffset,
                                       inputBytes, cudaMemcpyHostToDevice,
                                       stream));
  CUDA_REPORT_IF_ERROR(cudaMemcpyAsync(deviceWeight, weight + weightOffset,
                                       weightBytes, cudaMemcpyHostToDevice,
                                       stream));
  CUDA_REPORT_IF_ERROR(cudaMemcpyAsync(deviceBias, bias + biasOffset, biasBytes,
                                       cudaMemcpyHostToDevice, stream));
  CUDA_REPORT_IF_ERROR(cudaStreamSynchronize(stream));

  if (std::strcmp(backend, "cublas") == 0)
    mini_cublas_linear_relu_f32(deviceInput, deviceWeight, deviceBias,
                                deviceOutput, m, n, k, stream);
  else
    mini_cuda_linear_relu_f32(deviceInput, deviceWeight, deviceBias,
                              deviceOutput, m, n, k, stream);

  CUDA_REPORT_IF_ERROR(cudaMemcpyAsync(output + outputOffset, deviceOutput,
                                       outputBytes, cudaMemcpyDeviceToHost,
                                       stream));
  CUDA_REPORT_IF_ERROR(cudaStreamSynchronize(stream));

  cudaFree(deviceInput);
  cudaFree(deviceWeight);
  cudaFree(deviceBias);
  cudaFree(deviceOutput);
  CUDA_REPORT_IF_ERROR(cudaStreamDestroy(stream));
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mini_cuda_linear_relu_f32_memref(
    float *inputBase, float *inputData, int64_t inputOffset, int64_t inputSize0,
    int64_t inputSize1, int64_t inputStride0, int64_t inputStride1,
    float *weightBase, float *weightData, int64_t weightOffset,
    int64_t weightSize0, int64_t weightSize1, int64_t weightStride0,
    int64_t weightStride1, float *biasBase, float *biasData,
    int64_t biasOffset, int64_t biasSize0, int64_t biasStride0,
    float *outputBase, float *outputData, int64_t outputOffset,
    int64_t outputSize0, int64_t outputSize1, int64_t outputStride0,
    int64_t outputStride1, int64_t m, int64_t n, int64_t k) {
  (void)inputBase;
  (void)inputSize0;
  (void)inputSize1;
  (void)inputStride0;
  (void)inputStride1;
  (void)weightBase;
  (void)weightSize0;
  (void)weightSize1;
  (void)weightStride0;
  (void)weightStride1;
  (void)biasBase;
  (void)biasSize0;
  (void)biasStride0;
  (void)outputBase;
  (void)outputSize0;
  (void)outputSize1;
  (void)outputStride0;
  (void)outputStride1;
  runLinearReluMemRef("cuda_hand", inputData, inputOffset, weightData,
                      weightOffset, biasData, biasOffset, outputData,
                      outputOffset, m, n, k);
}

extern "C" MINI_CUDA_WRAPPERS_EXPORT void
mini_cublas_linear_relu_f32_memref(
    float *inputBase, float *inputData, int64_t inputOffset, int64_t inputSize0,
    int64_t inputSize1, int64_t inputStride0, int64_t inputStride1,
    float *weightBase, float *weightData, int64_t weightOffset,
    int64_t weightSize0, int64_t weightSize1, int64_t weightStride0,
    int64_t weightStride1, float *biasBase, float *biasData,
    int64_t biasOffset, int64_t biasSize0, int64_t biasStride0,
    float *outputBase, float *outputData, int64_t outputOffset,
    int64_t outputSize0, int64_t outputSize1, int64_t outputStride0,
    int64_t outputStride1, int64_t m, int64_t n, int64_t k) {
  (void)inputBase;
  (void)inputSize0;
  (void)inputSize1;
  (void)inputStride0;
  (void)inputStride1;
  (void)weightBase;
  (void)weightSize0;
  (void)weightSize1;
  (void)weightStride0;
  (void)weightStride1;
  (void)biasBase;
  (void)biasSize0;
  (void)biasStride0;
  (void)outputBase;
  (void)outputSize0;
  (void)outputSize1;
  (void)outputStride0;
  (void)outputStride1;
  runLinearReluMemRef("cublas", inputData, inputOffset, weightData,
                      weightOffset, biasData, biasOffset, outputData,
                      outputOffset, m, n, k);
}

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
