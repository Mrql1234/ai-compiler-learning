# compiler-mlir GPU 性能监控方案

本文记录 `compiler-mlir` GPU 路线的性能监控工程。当前目标不是追求完整工业级 benchmark，而是把同一个 case 在编译器生成 kernel、手写 CUDA kernel、第三方库 backend 之间做可重复对照，并把运行结果和 Nsight 报告落盘到仓库中，便于后续在其他机器上直接查看。

## 目标

性能监控系统比较三类 backend：

- `mlir_nvvm`：由 `mini-compiler-gpu-runner` 走 `mini -> gpu -> nvvm -> fatbin -> ExecutionEngine` 的编译器生成路线
- `cuda_hand`：手写 CUDA kernel baseline
- `cublas`：cuBLAS SGEMM 加 CUDA bias/ReLU epilogue 的库 baseline

对比时统一使用：

- 同一个 case JSON
- 同一套 warmup/repeat 协议
- 同一套 correctness tolerance
- 同一套 JSON 输出格式
- 同一个 `perf_compare.py` 汇总入口

重点回答：

- 编译器生成 kernel 是否正确
- `mlir_nvvm` 和手写 CUDA 的差距
- `mlir_nvvm` 和 cuBLAS 库路线的差距
- 主要瓶颈在 compile、JIT engine、launch、memcpy、kernel execution 还是 wrapper runtime

## 入口文件

核心入口：

- `perf/cases/gpu_runner_demo.json`
  - 小型 demo case
  - 已接入 `mlir_nvvm`、`cuda_hand`、`cublas`
  - 适合 smoke test、Nsight Systems、Nsight Compute
- `perf/cases/linear_relu_f32_m1024_n1024_k1024.json`
  - 大型 `linear + relu` case
  - 已接入 `cuda_hand`、`cublas`
  - 适合观察真实 GPU 吞吐差距
- `scripts/perf_run.py`
  - 统一运行入口
  - 负责调度 backend、写 backend JSON、写 `summary.json`
- `scripts/perf_compare.py`
  - 读取 `summary.json`
  - 输出 correctness、median/mean latency、相对 gap
- `scripts/perf_profile_nsys.sh`
  - Nsight Systems 包装脚本
  - 用于查看 compile、engine_create、warmup、benchmark、module_load、kernel_launch 等 NVTX range
- `scripts/perf_profile_ncu.sh`
  - Nsight Compute 包装脚本
  - 用于查看 kernel 级 occupancy、memory throughput、stall reason 等指标
- `tools/mini-compiler-gpu-runner.cpp`
  - `mlir_nvvm` backend 主入口
- `tools/mini-compiler-kernel-bench.cpp`
  - 外部 CUDA benchmark 主入口
- `tools/KernelBenchCuda.cu`
  - `cuda_hand` 和 `cublas` 的 CUDA 实现

已归档结果入口：

- `perf/runs/gpu_runner_demo_a10_20260511/summary.json`
- `perf/runs/gpu_runner_demo_a10_20260511/compare.txt`
- `perf/runs/linear_relu_f32_m1024_n1024_k1024_a10_20260511/summary.json`
- `perf/runs/linear_relu_f32_m1024_n1024_k1024_a10_20260511/compare.txt`
- `perf/profiles/a10_20260511/gpu_runner_demo_mlir_nvvm_nsys.nsys-rep`
- `perf/profiles/a10_20260511/gpu_runner_demo_mlir_nvvm_nsys_nvtx_summary.txt`
- `perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu.ncu-rep`
- `perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu_details.txt`
- `perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu_session.txt`

## 构建命令

在 A10 云 GPU 环境中使用：

```bash
cmake -S . -B build \
  -G Ninja \
  -DLLVM_DIR=/home/ql/toolchains/llvm_clang_static_analyzer/build/lib/cmake/llvm \
  -DMLIR_DIR=/home/ql/toolchains/llvm_clang_static_analyzer/build/lib/cmake/mlir \
  -DMINI_CUDA_ARCHITECTURES=86
cmake --build build -j2
```

`MINI_CUDA_ARCHITECTURES=86` 对应 NVIDIA A10 的 `sm_86`。

## 运行命令

运行小型 demo 的三 backend 对比：

```bash
python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend mlir_nvvm \
  --backend cuda_hand \
  --backend cublas \
  --warmup 10 \
  --repeat 50 \
  --run-dir perf/runs/gpu_runner_demo_a10_20260511
```

查看小型 demo 对比表：

```bash
python3 ./scripts/perf_compare.py \
  perf/runs/gpu_runner_demo_a10_20260511/summary.json
```

运行大型 `linear + relu` 的 CUDA/cuBLAS 对比：

```bash
python3 ./scripts/perf_run.py \
  perf/cases/linear_relu_f32_m1024_n1024_k1024.json \
  --warmup 10 \
  --repeat 50 \
  --run-dir perf/runs/linear_relu_f32_m1024_n1024_k1024_a10_20260511
```

查看大型 case 对比表：

```bash
python3 ./scripts/perf_compare.py \
  perf/runs/linear_relu_f32_m1024_n1024_k1024_a10_20260511/summary.json
```

直接运行手写 CUDA benchmark：

```bash
./build/bin/mini-compiler-kernel-bench \
  --backend cuda_hand \
  --case perf/cases/gpu_runner_demo.json \
  --warmup 10 \
  --repeat 50 \
  --json-output /tmp/cuda_hand.json
```

直接运行 cuBLAS benchmark：

```bash
./build/bin/mini-compiler-kernel-bench \
  --backend cublas \
  --case perf/cases/linear_relu_f32_m1024_n1024_k1024.json \
  --warmup 10 \
  --repeat 50 \
  --json-output /tmp/cublas.json
```

## Nsight 命令

采集 `mlir_nvvm` 路线的 Nsight Systems 报告：

```bash
./scripts/perf_profile_nsys.sh \
  perf/profiles/a10_20260511/gpu_runner_demo_mlir_nvvm_nsys \
  ./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
    --warmup=1 \
    --repeat=2 \
    --cubin-format=fatbin
```

导出 NVTX 文本摘要：

```bash
nsys stats --force-export=true \
  --report nvtx_pushpop_sum \
  perf/profiles/a10_20260511/gpu_runner_demo_mlir_nvvm_nsys.nsys-rep \
  | tee perf/profiles/a10_20260511/gpu_runner_demo_mlir_nvvm_nsys_nvtx_summary.txt
```

采集手写 CUDA kernel 的 Nsight Compute 报告：

```bash
./scripts/perf_profile_ncu.sh \
  perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu \
  ./build/bin/mini-compiler-kernel-bench \
    --backend cuda_hand \
    --case perf/cases/gpu_runner_demo.json \
    --warmup 1 \
    --repeat 1
```

导出 Nsight Compute 文本详情：

```bash
ncu --import perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu.ncu-rep \
  --page details \
  | tee perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu_details.txt

ncu --import perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu.ncu-rep \
  --page session \
  | tee perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu_session.txt
```

如果 `ncu` 报 `ERR_NVGPUCTRPERM`，说明当前用户没有 GPU performance counters 权限。当前机器已通过驱动参数确认：

```bash
grep RmProfilingAdminOnly /proc/driver/nvidia/params
```

期望输出：

```text
RmProfilingAdminOnly: 0
```

## 当前 A10 结果摘要

`gpu_runner_demo` 的归档结果：

```text
case: gpu_runner_demo
baseline: cublas
backend        correct   median ms   mean ms    gap
-------------  --------  ----------  --------  ------
mlir_nvvm      yes            0.479     0.607  40.69x
cuda_hand      yes            0.006     0.006   0.54x
cublas         yes            0.012     0.012   1.00x
```

`linear_relu_f32_m1024_n1024_k1024` 的归档结果：

```text
case: linear_relu_f32_m1024_n1024_k1024
baseline: cublas
backend        correct   median ms   mean ms    gap
-------------  --------  ----------  --------  ------
cuda_hand      yes            5.109     5.109  30.63x
cublas         yes            0.167     0.167   1.00x
```

Nsight Systems 的 NVTX 摘要显示 `mlir_nvvm` 当前主要时间集中在：

- `warmup`
- `module_load`
- `compile`
- `engine_create`
- `benchmark`
- `kernel_launch`

Nsight Compute 的手写 CUDA demo 结论是：小型 demo grid 过小，无法填满 A10 的 72 个 SM，报告中可以看到低 occupancy、低 SM throughput，以及 grid size 太小的优化提示。这符合该 demo 作为 smoke test 的定位；大型 case 更适合作为吞吐性能对比。

## 后续优化循环

每次优化建议按下面流程执行：

1. 用 `perf_run.py` 跑所有可用 backend。
2. 用 `perf_compare.py` 看 correctness 和 gap。
3. 用 Nsight Systems 判断瓶颈属于 compile/JIT/runtime/launch/memcpy/kernel 哪一类。
4. 用 Nsight Compute 看 kernel 内部瓶颈。
5. 修改 lowering、runtime wrapper、kernel mapping、手写 CUDA 或库 backend 配置。
6. 重新跑同一个 case。
7. 将稳定结果保存到 `perf/runs/` 和 `perf/profiles/`。
