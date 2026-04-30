# Cloud GPU Development Workflow

This repository is intended to be developed from two machines:

- Local laptop/WSL: edit code, run CPU-only Python tests, read docs, commit changes.
- Alibaba Cloud GPU ECS: run CUDA/Triton/vLLM/SGLang/TensorRT experiments and MLIR/CUDA builds.

## Recommended Cloud Instance

- Instance family: `gn7i`
- Example size: `ecs.gn7i-c8g1.2xlarge`
- GPU: NVIDIA A10 24 GiB
- OS image: Ubuntu 22.04 64-bit with NVIDIA GPU Driver
- System disk: ESSD 200 GiB or larger
- Billing: pay-as-you-go; stop with economical mode when idle

Verify the GPU after login:

```bash
nvidia-smi
```

Expected result: one NVIDIA A10 GPU is visible.

## Base System Setup

```bash
apt update
apt install -y \
  git curl wget ca-certificates \
  build-essential cmake ninja-build clang lld \
  python3 python3-venv python3-pip \
  pkg-config
```

Create a non-root development user:

```bash
adduser ql
usermod -aG sudo ql
```

Then reconnect as `ql`.

## Clone the Repository

```bash
mkdir -p ~/code
cd ~/code
git clone git@github.com:Mrql1234/ai-compiler-learning.git
cd ai-compiler-learning
```

If SSH keys are not configured on the cloud machine, use HTTPS first:

```bash
git clone https://github.com/Mrql1234/ai-compiler-learning.git
```

## Python Project Setup

The CPU-only mini compiler can run on either local WSL or the cloud machine.

```bash
cd ~/code/ai-compiler-learning/projects/mini-ai-compiler
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pytest
```

Install PyTorch depending on where the code runs:

```bash
# CPU-only local/WSL setup
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

On the cloud GPU machine, install the CUDA-enabled PyTorch wheel that matches the current PyTorch installation guide. Then verify:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## Python Validation

```bash
cd ~/code/ai-compiler-learning/projects/mini-ai-compiler
source .venv/bin/activate
python -m unittest discover -s tests
python -m tools.run_mlp_example
python -m tools.dump_ir
```

## MLIR Pass Project Build

This project expects an existing LLVM/MLIR build or installation.

```bash
cd ~/code/ai-compiler-learning/projects/mlir-passes
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm
cmake --build build
```

Run an example pass after building:

```bash
./build/bin/mlir-passes-opt test/constant_fold.mlir
```

## Mini Compiler MLIR Skeleton Build

```bash
cd ~/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm
cmake --build build
```

Smoke run:

```bash
./build/bin/mini-compiler-opt test/smoke.mlir
```

## GPU-Specific Work

Use the cloud GPU machine for:

- CUDA and Triton kernels
- vLLM and SGLang runtime experiments
- TensorRT experiments
- GPU benchmark runs

Keep large generated artifacts, downloaded models, and build directories out of git. Prefer:

- Git for source code and small docs
- OSS or a dedicated data disk for models/datasets
- Snapshots or custom images for expensive-to-recreate environments

## Daily Workflow

Local laptop:

```bash
git pull
# edit code
git add .
git commit -m "..."
git push
```

Cloud GPU machine:

```bash
cd ~/code/ai-compiler-learning
git pull
# run GPU-specific build/test/benchmark
```

When idle, stop the ECS instance from the Alibaba Cloud console using economical mode. Do not release the instance unless the system disk and important data have been backed up.
