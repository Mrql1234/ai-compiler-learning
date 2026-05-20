#include "KernelBench.h"

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MINI_ENABLE_NVTX
#include "nvtx3/nvToolsExt.h"
#endif

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

static void checkCuda(cudaError_t status, const char *expr) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expr) + " failed with " +
                             cudaGetErrorString(status));
  }
}

static void checkCublas(cublasStatus_t status, const char *expr) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(expr) + " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
}

#define CHECK_CUDA(expr) checkCuda((expr), #expr)
#define CHECK_CUBLAS(expr) checkCublas((expr), #expr)

__global__ void linearReluKernel(const float *input, const float *weight,
                                 const float *bias, float *output, int m, int n,
                                 int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n)
    return;

  float acc = bias[col];
  for (int kk = 0; kk < k; ++kk)
    acc += input[row * k + kk] * weight[col * k + kk];
  output[row * n + col] = fmaxf(acc, 0.0f);
}

__global__ void biasReluKernel(float *output, const float *bias, int m, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = m * n;
  if (index >= total)
    return;
  int col = index % n;
  output[index] = fmaxf(output[index] + bias[col], 0.0f);
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

static void fillInputs(const BenchProblem &problem, std::vector<float> &input,
                       std::vector<float> &weight,
                       std::vector<float> &bias) {
  std::fill(input.begin(), input.end(), 0.0f);
  std::fill(weight.begin(), weight.end(), 0.0f);
  std::fill(bias.begin(), bias.end(), 0.0f);

  if (problem.dataProfile == "gpu_runner_demo") {
    if (problem.m != 2 || problem.n != 8 || problem.k != 4)
      throw std::runtime_error(
          "gpu_runner_demo data profile expects m=2, n=8, k=4");
    input[0] = 1.0f;
    input[1] = 1.0f;
    weight[0] = 1.0f;
    weight[1] = 2.0f;
    bias[0] = 0.5f;
    return;
  }

  for (int64_t row = 0; row < problem.m; ++row)
    for (int64_t col = 0; col < problem.k; ++col)
      input[row * problem.k + col] = deterministicInput(row, col);
  for (int64_t row = 0; row < problem.n; ++row)
    for (int64_t col = 0; col < problem.k; ++col)
      weight[row * problem.k + col] = deterministicWeight(row, col);
  for (int64_t col = 0; col < problem.n; ++col)
    bias[col] = deterministicBias(col);
}

struct DeviceBuffers {
  float *input = nullptr;
  float *weight = nullptr;
  float *bias = nullptr;
  float *output = nullptr;
};

static DeviceBuffers allocateAndCopy(const BenchProblem &problem,
                                     cudaStream_t stream) {
  NvtxRange range("bench_copy_inputs");
  std::vector<float> input(problem.m * problem.k);
  std::vector<float> weight(problem.n * problem.k);
  std::vector<float> bias(problem.n);
  fillInputs(problem, input, weight, bias);

  DeviceBuffers buffers;
  CHECK_CUDA(cudaMalloc(&buffers.input, input.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&buffers.weight, weight.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&buffers.bias, bias.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&buffers.output,
                        static_cast<size_t>(problem.m * problem.n) *
                            sizeof(float)));
  CHECK_CUDA(cudaMemcpyAsync(buffers.input, input.data(),
                             input.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(buffers.weight, weight.data(),
                             weight.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(buffers.bias, bias.data(),
                             bias.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return buffers;
}

static void freeBuffers(DeviceBuffers &buffers) {
  cudaFree(buffers.input);
  cudaFree(buffers.weight);
  cudaFree(buffers.bias);
  cudaFree(buffers.output);
  buffers = {};
}

static float copyFirstResult(DeviceBuffers &buffers, cudaStream_t stream) {
  NvtxRange range("bench_copyback");
  float result = 0.0f;
  CHECK_CUDA(cudaMemcpyAsync(&result, buffers.output, sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return result;
}

template <typename LaunchFn>
static void timeRepeated(int warmup, int repeat, cudaStream_t stream,
                         LaunchFn launch, BenchResult &result) {
  result.invokeTimingsMs.reserve(static_cast<size_t>(repeat));
  result.kernelTimingsMs.reserve(static_cast<size_t>(repeat));
  cudaEvent_t startEvent = nullptr;
  cudaEvent_t stopEvent = nullptr;
  CHECK_CUDA(cudaEventCreate(&startEvent));
  CHECK_CUDA(cudaEventCreate(&stopEvent));
  for (int iteration = 0; iteration < warmup + repeat; ++iteration) {
    auto start = std::chrono::steady_clock::now();
    CHECK_CUDA(cudaEventRecord(startEvent, stream));
    launch();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stopEvent, stream));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    auto end = std::chrono::steady_clock::now();
    if (iteration >= warmup) {
      float kernelMs = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&kernelMs, startEvent, stopEvent));
      std::chrono::duration<double, std::milli> elapsed = end - start;
      result.invokeTimingsMs.push_back(elapsed.count());
      result.kernelTimingsMs.push_back(static_cast<double>(kernelMs));
    }
  }
  CHECK_CUDA(cudaEventDestroy(startEvent));
  CHECK_CUDA(cudaEventDestroy(stopEvent));
  result.timingsMs = result.invokeTimingsMs;
}

} // namespace

BenchResult runCudaHandBenchmark(const BenchProblem &problem, int warmup,
                                 int repeat) {
  if (problem.operation != "linear_relu")
    throw std::runtime_error("cuda_hand only supports operation=linear_relu");

  NvtxRange range("cuda_hand_total");
  CHECK_CUDA(cudaSetDevice(0));
  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceBuffers buffers = allocateAndCopy(problem, stream);

  dim3 block(16, 16);
  dim3 grid((static_cast<unsigned>(problem.n) + block.x - 1) / block.x,
            (static_cast<unsigned>(problem.m) + block.y - 1) / block.y);
  BenchResult benchResult;
  benchResult.backend = "cuda_hand";
  benchResult.implementation = "single CUDA linear+relu kernel";

  timeRepeated(warmup, repeat, stream, [&] {
    NvtxRange launchRange("cuda_hand_linear_relu");
    linearReluKernel<<<grid, block, 0, stream>>>(
        buffers.input, buffers.weight, buffers.bias, buffers.output,
        static_cast<int>(problem.m), static_cast<int>(problem.n),
        static_cast<int>(problem.k));
  }, benchResult);

  float result = copyFirstResult(buffers, stream);
  freeBuffers(buffers);
  CHECK_CUDA(cudaStreamDestroy(stream));

  benchResult.result = result;
  return benchResult;
}

BenchResult runCublasBenchmark(const BenchProblem &problem, int warmup,
                               int repeat) {
  if (problem.operation != "linear_relu")
    throw std::runtime_error("cublas only supports operation=linear_relu");

  NvtxRange range("cublas_total");
  CHECK_CUDA(cudaSetDevice(0));
  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetStream(handle, stream));

  DeviceBuffers buffers = allocateAndCopy(problem, stream);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int m = static_cast<int>(problem.m);
  int n = static_cast<int>(problem.n);
  int k = static_cast<int>(problem.k);
  int total = m * n;
  int block = 256;
  int grid = (total + block - 1) / block;

  BenchResult benchResult;
  benchResult.backend = "cublas";
  benchResult.implementation = "cuBLAS SGEMM + CUDA bias/relu kernel";

  timeRepeated(warmup, repeat, stream, [&] {
    NvtxRange launchRange("cublas_sgemm_bias_relu");
    // Row-major C[M,N] is column-major C^T[N,M]. With row-major A[M,K]
    // and W[N,K], use column-major views A_col[K,M] and W_col[K,N].
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                             &alpha, buffers.weight, k, buffers.input, k,
                             &beta, buffers.output, n));
    biasReluKernel<<<grid, block, 0, stream>>>(buffers.output, buffers.bias, m,
                                               n);
  }, benchResult);

  float result = copyFirstResult(buffers, stream);
  freeBuffers(buffers);
  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaStreamDestroy(stream));

  benchResult.result = result;
  return benchResult;
}
