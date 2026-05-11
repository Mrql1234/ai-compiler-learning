#include "cuda_lab/cuda_check.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

__global__ void reduce_sum_kernel(const float* input, float* partial, int n) {
  __shared__ float shared[256];

  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  float value = 0.0f;
  if (index < n) {
    value = input[index];
  }
  shared[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial[blockIdx.x] = shared[0];
  }
}

float run_reduce(const float* d_input, int n) {
  constexpr int threads = 256;
  int current_n = n;

  float* current_input = const_cast<float*>(d_input);
  float* d_partial = nullptr;
  std::vector<float*> buffers;

  while (current_n > 1) {
    int blocks = (current_n + threads - 1) / threads;
    CUDA_CHECK(cudaMalloc(&d_partial, static_cast<size_t>(blocks) * sizeof(float)));
    buffers.push_back(d_partial);
    reduce_sum_kernel<<<blocks, threads>>>(current_input, d_partial, current_n);
    CUDA_CHECK(cudaGetLastError());
    current_input = d_partial;
    current_n = blocks;
  }

  float result = 0.0f;
  CUDA_CHECK(cudaMemcpy(&result, current_input, sizeof(float), cudaMemcpyDeviceToHost));

  for (float* buffer : buffers) {
    CUDA_CHECK(cudaFree(buffer));
  }
  return result;
}

}  // namespace

int main() {
  const int n = 1 << 20;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  std::vector<float> h_input(n);
  for (int i = 0; i < n; ++i) {
    h_input[i] = 1.0f + static_cast<float>(i % 7) * 0.1f;
  }

  double cpu_ref = std::accumulate(h_input.begin(), h_input.end(), 0.0);

  float* d_input = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  float gpu_result = run_reduce(d_input, n);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  double abs_diff = std::fabs(static_cast<double>(gpu_result) - cpu_ref);
  double tolerance = std::fabs(cpu_ref) * 1e-7;
  if (tolerance < 1e-2) {
    tolerance = 1e-2;
  }

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "CUDA Reduce Sum Demo\n";
  std::cout << "n = " << n << "\n";
  std::cout << "gpu result = " << gpu_result << "\n";
  std::cout << "cpu result = " << cpu_ref << "\n";
  std::cout << "abs diff = " << abs_diff << "\n";
  std::cout << "tolerance = " << tolerance << "\n";
  std::cout << "elapsed = " << elapsed_ms << " ms\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_input));
  return abs_diff <= tolerance ? 0 : 1;
}
