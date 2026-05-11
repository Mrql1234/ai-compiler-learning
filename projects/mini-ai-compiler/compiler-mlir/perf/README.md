# compiler-mlir 性能监控入口

`perf/` 目录保存可重复的性能 case、运行结果和 Nsight 报告。当前已归档一组 A10 云 GPU 上的真实执行数据，另一台机器拉取仓库后可以直接查看 JSON、文本摘要和 Nsight report 文件，不需要重新运行脚本。

## 入口文件

- `perf/cases/gpu_runner_demo.json`
  - 小型 `linear + relu` demo
  - backend：`mlir_nvvm`、`cuda_hand`、`cublas`
- `perf/cases/linear_relu_f32_m1024_n1024_k1024.json`
  - 大型 `linear + relu` case
  - backend：`cuda_hand`、`cublas`
- `scripts/perf_run.py`
  - 统一运行入口
  - 生成每个 backend 的 JSON 和 `summary.json`
- `scripts/perf_compare.py`
  - 读取 `summary.json`
  - 输出 correctness、latency、gap
- `scripts/perf_profile_nsys.sh`
  - Nsight Systems 包装入口
- `scripts/perf_profile_ncu.sh`
  - Nsight Compute 包装入口
- `tools/mini-compiler-kernel-bench.cpp`
  - 手写 CUDA 和 cuBLAS benchmark 入口
- `tools/KernelBenchCuda.cu`
  - CUDA kernel 和 cuBLAS 调用实现

## 已归档数据

小型 demo：

- `perf/runs/gpu_runner_demo_a10_20260511/summary.json`
- `perf/runs/gpu_runner_demo_a10_20260511/compare.txt`
- `perf/runs/gpu_runner_demo_a10_20260511/mlir_nvvm.json`
- `perf/runs/gpu_runner_demo_a10_20260511/cuda_hand.json`
- `perf/runs/gpu_runner_demo_a10_20260511/cublas.json`
- `perf/runs/gpu_runner_demo_a10_20260511/mlir_nvvm_lowered.mlir`

大型 case：

- `perf/runs/linear_relu_f32_m1024_n1024_k1024_a10_20260511/summary.json`
- `perf/runs/linear_relu_f32_m1024_n1024_k1024_a10_20260511/compare.txt`
- `perf/runs/linear_relu_f32_m1024_n1024_k1024_a10_20260511/cuda_hand.json`
- `perf/runs/linear_relu_f32_m1024_n1024_k1024_a10_20260511/cublas.json`

Nsight 报告：

- `perf/profiles/a10_20260511/gpu_runner_demo_mlir_nvvm_nsys.nsys-rep`
- `perf/profiles/a10_20260511/gpu_runner_demo_mlir_nvvm_nsys.sqlite`
- `perf/profiles/a10_20260511/gpu_runner_demo_mlir_nvvm_nsys_nvtx_summary.txt`
- `perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu.ncu-rep`
- `perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu_details.txt`
- `perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu_session.txt`

## 构建命令

```bash
cmake -S . -B build \
  -G Ninja \
  -DLLVM_DIR=/home/ql/toolchains/llvm_clang_static_analyzer/build/lib/cmake/llvm \
  -DMLIR_DIR=/home/ql/toolchains/llvm_clang_static_analyzer/build/lib/cmake/mlir \
  -DMINI_CUDA_ARCHITECTURES=86
cmake --build build -j2
```

## 运行命令

运行 `gpu_runner_demo` 的三 backend 对比：

```bash
python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend mlir_nvvm \
  --backend cuda_hand \
  --backend cublas \
  --warmup 10 \
  --repeat 50 \
  --run-dir perf/runs/gpu_runner_demo_a10_20260511
```

查看对比结果：

```bash
python3 ./scripts/perf_compare.py \
  perf/runs/gpu_runner_demo_a10_20260511/summary.json
```

运行大型 `linear + relu` case：

```bash
python3 ./scripts/perf_run.py \
  perf/cases/linear_relu_f32_m1024_n1024_k1024.json \
  --warmup 10 \
  --repeat 50 \
  --run-dir perf/runs/linear_relu_f32_m1024_n1024_k1024_a10_20260511
```

查看大型 case 对比结果：

```bash
python3 ./scripts/perf_compare.py \
  perf/runs/linear_relu_f32_m1024_n1024_k1024_a10_20260511/summary.json
```

直接运行 benchmark binary：

```bash
./build/bin/mini-compiler-kernel-bench \
  --backend cuda_hand \
  --case perf/cases/gpu_runner_demo.json \
  --warmup 10 \
  --repeat 50 \
  --json-output /tmp/cuda_hand.json

./build/bin/mini-compiler-kernel-bench \
  --backend cublas \
  --case perf/cases/linear_relu_f32_m1024_n1024_k1024.json \
  --warmup 10 \
  --repeat 50 \
  --json-output /tmp/cublas.json
```

## Nsight 命令

采集 Nsight Systems：

```bash
./scripts/perf_profile_nsys.sh \
  perf/profiles/a10_20260511/gpu_runner_demo_mlir_nvvm_nsys \
  ./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
    --warmup=1 \
    --repeat=2 \
    --cubin-format=fatbin
```

导出 NVTX 摘要：

```bash
nsys stats --force-export=true \
  --report nvtx_pushpop_sum \
  perf/profiles/a10_20260511/gpu_runner_demo_mlir_nvvm_nsys.nsys-rep \
  | tee perf/profiles/a10_20260511/gpu_runner_demo_mlir_nvvm_nsys_nvtx_summary.txt
```

采集 Nsight Compute：

```bash
./scripts/perf_profile_ncu.sh \
  perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu \
  ./build/bin/mini-compiler-kernel-bench \
    --backend cuda_hand \
    --case perf/cases/gpu_runner_demo.json \
    --warmup 1 \
    --repeat 1
```

导出 Nsight Compute 文本报告：

```bash
ncu --import perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu.ncu-rep \
  --page details \
  | tee perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu_details.txt

ncu --import perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu.ncu-rep \
  --page session \
  | tee perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu_session.txt
```

## 当前 A10 结果

`gpu_runner_demo`：

```text
case: gpu_runner_demo
baseline: cublas
backend        correct   median ms   mean ms    gap
-------------  --------  ----------  --------  ------
mlir_nvvm      yes            0.479     0.607  40.69x
cuda_hand      yes            0.006     0.006   0.54x
cublas         yes            0.012     0.012   1.00x
```

`linear_relu_f32_m1024_n1024_k1024`：

```text
case: linear_relu_f32_m1024_n1024_k1024
baseline: cublas
backend        correct   median ms   mean ms    gap
-------------  --------  ----------  --------  ------
cuda_hand      yes            5.109     5.109  30.63x
cublas         yes            0.167     0.167   1.00x
```
