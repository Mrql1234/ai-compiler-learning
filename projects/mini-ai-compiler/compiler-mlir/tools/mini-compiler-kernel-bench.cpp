#include "KernelBench.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string backend = "cuda_hand";
  std::string casePath;
  std::string jsonOutput;
  int warmup = 10;
  int repeat = 100;
};

static void printUsage(llvm::raw_ostream &os, llvm::StringRef programName) {
  os << "Usage: " << programName
     << " --backend cuda_hand|cublas|cutlass --case perf/cases/name.json"
        " [--warmup N] [--repeat N] [--json-output path]\n";
}

static bool parseInt(llvm::StringRef value, llvm::StringRef optionName,
                     int &result) {
  int parsed = 0;
  if (value.getAsInteger(10, parsed) || parsed < 0) {
    llvm::errs() << "Expected " << optionName
                 << " to be a non-negative integer\n";
    return false;
  }
  result = parsed;
  return true;
}

static bool parseOptions(int argc, char **argv, Options &options,
                         bool &printedHelp) {
  for (int index = 1; index < argc; ++index) {
    llvm::StringRef arg(argv[index]);
    auto consumeValue = [&](llvm::StringRef name,
                            std::string &target) -> bool {
      if (arg == name) {
        if (index + 1 >= argc) {
          llvm::errs() << "Missing value after " << name << "\n";
          return false;
        }
        target = argv[++index];
        return true;
      }
      if (arg.consume_front((name + "=").str())) {
        target = arg.str();
        return true;
      }
      return false;
    };

    if (arg == "-h" || arg == "--help") {
      printUsage(llvm::outs(), argv[0]);
      printedHelp = true;
      return false;
    }
    if (consumeValue("--backend", options.backend) ||
        consumeValue("--case", options.casePath) ||
        consumeValue("--json-output", options.jsonOutput))
      continue;
    if (arg == "--warmup") {
      if (index + 1 >= argc ||
          !parseInt(argv[++index], "--warmup", options.warmup))
        return false;
      continue;
    }
    if (arg.consume_front("--warmup=")) {
      if (!parseInt(arg, "--warmup", options.warmup))
        return false;
      continue;
    }
    if (arg == "--repeat") {
      if (index + 1 >= argc ||
          !parseInt(argv[++index], "--repeat", options.repeat))
        return false;
      if (options.repeat == 0) {
        llvm::errs() << "Expected --repeat to be greater than zero\n";
        return false;
      }
      continue;
    }
    if (arg.consume_front("--repeat=")) {
      if (!parseInt(arg, "--repeat", options.repeat))
        return false;
      if (options.repeat == 0) {
        llvm::errs() << "Expected --repeat to be greater than zero\n";
        return false;
      }
      continue;
    }
    llvm::errs() << "Unknown option: " << arg << "\n";
    return false;
  }

  if (options.casePath.empty()) {
    llvm::errs() << "Missing --case\n";
    printUsage(llvm::errs(), argv[0]);
    return false;
  }
  return true;
}

static std::string takeErrorMessage(llvm::Error error) {
  std::string message;
  llvm::raw_string_ostream os(message);
  llvm::logAllUnhandledErrors(std::move(error), os);
  return os.str();
}

static BenchProblem loadProblem(llvm::StringRef casePath) {
  auto buffer = llvm::MemoryBuffer::getFile(casePath);
  if (!buffer)
    throw std::runtime_error("failed to read case file '" + casePath.str() +
                             "': " + buffer.getError().message());

  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    throw std::runtime_error("failed to parse case JSON: " +
                             takeErrorMessage(parsed.takeError()));

  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    throw std::runtime_error("case JSON root must be an object");
  const llvm::json::Object *problemObject = root->getObject("problem");
  if (!problemObject)
    throw std::runtime_error("case JSON must define a problem object");

  BenchProblem problem;
  if (std::optional<llvm::StringRef> operation =
          problemObject->getString("operation"))
    problem.operation = operation->str();
  if (std::optional<llvm::StringRef> dataProfile =
          problemObject->getString("data_profile"))
    problem.dataProfile = dataProfile->str();
  if (std::optional<int64_t> m = problemObject->getInteger("m"))
    problem.m = *m;
  if (std::optional<int64_t> n = problemObject->getInteger("n"))
    problem.n = *n;
  if (std::optional<int64_t> k = problemObject->getInteger("k"))
    problem.k = *k;

  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0)
    throw std::runtime_error("problem dimensions must be positive");
  return problem;
}

static double latencyMin(const std::vector<double> &values) {
  return values.empty() ? 0.0 : *std::min_element(values.begin(), values.end());
}

static double latencyMax(const std::vector<double> &values) {
  return values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
}

static double latencyMean(const std::vector<double> &values) {
  if (values.empty())
    return 0.0;
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

static double latencyMedian(std::vector<double> values) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  size_t middle = values.size() / 2;
  if (values.size() % 2 == 1)
    return values[middle];
  return (values[middle - 1] + values[middle]) / 2.0;
}

static void writeJsonString(llvm::raw_ostream &os, llvm::StringRef value) {
  os << "\"";
  for (char c : value) {
    switch (c) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      os << c;
      break;
    }
  }
  os << "\"";
}

static void writeJsonReport(const Options &options, const BenchProblem &problem,
                            const BenchResult &result) {
  if (options.jsonOutput.empty())
    return;

  std::error_code error;
  llvm::raw_fd_ostream os(options.jsonOutput, error);
  if (error)
    throw std::runtime_error("failed to open JSON output path '" +
                             options.jsonOutput + "': " + error.message());

  os << "{\n";
  os << "  \"backend\": ";
  writeJsonString(os, options.backend);
  os << ",\n";
  os << "  \"kind\": \"external_cuda\",\n";
  os << "  \"implementation\": ";
  writeJsonString(os, result.implementation);
  os << ",\n";
  os << "  \"problem\": {\n";
  os << "    \"operation\": ";
  writeJsonString(os, problem.operation);
  os << ",\n";
  os << "    \"data_profile\": ";
  writeJsonString(os, problem.dataProfile);
  os << ",\n";
  os << "    \"m\": " << problem.m << ",\n";
  os << "    \"n\": " << problem.n << ",\n";
  os << "    \"k\": " << problem.k << "\n";
  os << "  },\n";
  os << "  \"warmup\": " << options.warmup << ",\n";
  os << "  \"repeat\": " << options.repeat << ",\n";
  os << "  \"result\": " << result.result << ",\n";
  os << "  \"latency_ms\": {\n";
  os << "    \"min\": " << latencyMin(result.timingsMs) << ",\n";
  os << "    \"mean\": " << latencyMean(result.timingsMs) << ",\n";
  os << "    \"median\": " << latencyMedian(result.timingsMs) << ",\n";
  os << "    \"max\": " << latencyMax(result.timingsMs) << "\n";
  os << "  },\n";
  os << "  \"timings_ms\": [";
  for (size_t i = 0, e = result.timingsMs.size(); i < e; ++i) {
    if (i != 0)
      os << ", ";
    os << result.timingsMs[i];
  }
  os << "],\n";
  os << "  \"artifacts\": {\n";
  os << "    \"timing_mode\": \"host_wall_launch_to_sync\"\n";
  os << "  }\n";
  os << "}\n";
}

static BenchResult runSelectedBackend(const Options &options,
                                      const BenchProblem &problem) {
  if (options.backend == "cuda_hand")
    return runCudaHandBenchmark(problem, options.warmup, options.repeat);
  if (options.backend == "cublas")
    return runCublasBenchmark(problem, options.warmup, options.repeat);
  if (options.backend == "cutlass") {
    BenchResult result =
        runCublasBenchmark(problem, options.warmup, options.repeat);
    result.backend = "cutlass";
    result.implementation = "cuBLAS fallback for library baseline";
    return result;
  }
  throw std::runtime_error("unsupported backend: " + options.backend);
}

} // namespace

int main(int argc, char **argv) {
  Options options;
  bool printedHelp = false;
  if (!parseOptions(argc, argv, options, printedHelp))
    return printedHelp ? 0 : 1;

  try {
    BenchProblem problem = loadProblem(options.casePath);
    BenchResult result = runSelectedBackend(options, problem);
    writeJsonReport(options, problem, result);
    llvm::outs() << result.result << "\n";
    return 0;
  } catch (const std::exception &exception) {
    llvm::errs() << "mini-compiler-kernel-bench: " << exception.what() << "\n";
    return 1;
  }
}
