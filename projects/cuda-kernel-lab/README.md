# CUDA Kernel Lab

`CUDA Kernel Lab` 是一个独立于 `triton-kernel-library` 的手写 CUDA 学习项目，目标是把常见算子的实现、benchmark 和 Nsight 性能分析放到同一个最小实验场里。

这个项目强调三件事：

- 手写 CUDA kernel
- 和 CPU / PyTorch 参考实现做正确性与性能对比
- 用 `nsys` / `ncu` 观察 launch、occupancy、memory throughput 等指标

## 项目定位

这个项目不是一个通用深度学习框架，也不是生产级算子库。

当前更适合作为：

- CUDA 算子实习仓库
- 面试 / 简历项目素材
- Triton 与手写 CUDA 的对照实验区
- Nsight Systems / Nsight Compute 的练习入口

## 当前目录结构

```text
projects/cuda-kernel-lab/
  include/cuda_lab/
    cuda_check.h
  src/
    vector_add.cu
    reduce_sum.cu
  scripts/
    build.sh
    run_vector_add.sh
    run_reduce_sum.sh
    profile_nsys_vector_add.sh
    profile_ncu_vector_add.sh
  CMakeLists.txt
  requirements.md
  design.md
  tasks.md
  README.md
```

## 当前入口文件

- `src/vector_add.cu`
  - 入口程序：`cuda_vector_add`
  - 作用：最小可运行 CUDA kernel 示例，包含正确性校验和简单 benchmark
- `src/reduce_sum.cu`
  - 入口程序：`cuda_reduce_sum`
  - 作用：展示 block 内归约与多轮 partial reduction 的基础写法

## 构建命令

```bash
cd projects/cuda-kernel-lab
bash scripts/build.sh
```

如果你想手动执行：

```bash
cd projects/cuda-kernel-lab
cmake -S . -B build -G Ninja
cmake --build build -j
```

默认构建脚本会通过 `nvidia-smi` 探测本机 GPU compute capability，并传给 CMake。也可以手动覆盖：

```bash
cd projects/cuda-kernel-lab
CUDA_ARCHITECTURES=86 bash scripts/build.sh
```

如果 CMake 提示找不到 `nvcc`，通常说明当前环境没有正确暴露 CUDA Toolkit，可以显式指定：

```bash
cd projects/cuda-kernel-lab
cmake -S . -B build -G Ninja -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build build -j
```

## 运行命令

运行 `vector add`：

```bash
cd projects/cuda-kernel-lab
bash scripts/run_vector_add.sh
```

运行 `reduce sum`：

```bash
cd projects/cuda-kernel-lab
bash scripts/run_reduce_sum.sh
```

## Nsight 分析命令

使用 `nsys` 看时间线和 kernel 启动：

```bash
cd projects/cuda-kernel-lab
bash scripts/profile_nsys_vector_add.sh
```

使用 `ncu` 看吞吐、occupancy 和 memory 指标：

```bash
cd projects/cuda-kernel-lab
bash scripts/profile_ncu_vector_add.sh
```

## 环境要求

- NVIDIA GPU
- CUDA Toolkit（需要 `nvcc`）
- CMake 3.20+
- Ninja

可以先用下面命令自检：

```bash
nvcc --version
cmake --version
ninja --version
```

## 后续建议的算子路线

建议按下面顺序推进，避免一上来就做过重的 kernel：

1. `vector add`
2. `reduce sum`
3. `softmax`
4. `layernorm`
5. `sgemm`
6. `conv2d`
7. `rope`
8. `attention`

每加一个算子，最好同时补三类东西：

- 一个可运行入口
- 一个 benchmark
- 一份 profiling 记录或结论

## 和 `triton-kernel-library` 的关系

- `projects/triton-kernel-library/`：偏 Triton 实现与 Python 侧实验
- `projects/cuda-kernel-lab/`：偏手写 CUDA、CMake 构建与 Nsight 分析

这样拆开后，两个项目各自边界更清晰，但后续可以按相同算子做横向对照。
