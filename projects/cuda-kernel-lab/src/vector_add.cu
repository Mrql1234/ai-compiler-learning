#include "cuda_lab/cuda_check.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    c[index] = a[index] + b[index];
  }
}

float benchmark_kernel(const float* d_a, const float* d_b, float* d_c, int n, int runs) {
  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < 10; ++i) {
    vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < runs; ++i) {
    vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / static_cast<float>(runs);
}

}  // namespace

int main() {
  const int n = 1 << 24;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  std::vector<float> h_a(n);
  std::vector<float> h_b(n);
  std::vector<float> h_c(n, 0.0f);
  std::vector<float> h_ref(n, 0.0f);

  for (int i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i % 1024) * 0.5f;
    h_b[i] = static_cast<float>(i % 257) * 0.25f;
    h_ref[i] = h_a[i] + h_b[i];
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  float avg_ms = benchmark_kernel(d_a, d_b, d_c, n, 100);

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  float max_diff = 0.0f;
  for (int i = 0; i < n; ++i) {
    max_diff = std::max(max_diff, std::fabs(h_c[i] - h_ref[i]));
  }

  double bandwidth_gb_s = (3.0 * static_cast<double>(bytes) / 1e9) / (avg_ms / 1e3);

  std::cout << "CUDA Vector Add Demo\n";
  std::cout << "n = " << n << "\n";
  std::cout << "avg time = " << avg_ms << " ms\n";
  std::cout << "estimated bandwidth = " << bandwidth_gb_s << " GB/s\n";
  std::cout << "max diff = " << max_diff << "\n";

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return max_diff < 1e-6f ? 0 : 1;
}

