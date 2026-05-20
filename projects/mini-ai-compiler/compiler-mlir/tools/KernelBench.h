#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct BenchProblem {
  std::string operation = "linear_relu";
  std::string dataProfile = "deterministic";
  int64_t m = 2;
  int64_t n = 8;
  int64_t k = 4;
};

struct BenchResult {
  std::string backend;
  std::string implementation;
  float result = 0.0f;
  std::vector<double> invokeTimingsMs;
  std::vector<double> kernelTimingsMs;
  std::vector<double> timingsMs;
};

BenchResult runCudaHandBenchmark(const BenchProblem &problem, int warmup,
                                 int repeat);
BenchResult runCublasBenchmark(const BenchProblem &problem, int warmup,
                               int repeat);
