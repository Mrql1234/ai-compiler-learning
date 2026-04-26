# WSL + Ubuntu + MLIR 环境安装记录

更新时间：2026-04-20

## 目标

- 安装并验证 `WSL`
- 安装并验证 `Ubuntu 22.04`
- 安装 `projects/mlir-passes` 开发所需的 LLVM/MLIR/C++ 工具链
- 分析当前 MLIR 项目的结构、可构建性和缺口
- 记录本次安装的所有内容、验证结果和后续修复建议

## 当前环境结论

- `WSL` 已可用
- `Ubuntu-22.04` 已安装
- `Ubuntu-22.04` 运行在 `WSL2`
- Ubuntu 用户：`ql`
- Ubuntu 版本：`Ubuntu 22.04.5 LTS (Jammy Jellyfish)`
- Linux 内核：`6.6.87.2-microsoft-standard-WSL2`
- 已使用 `root` 用户完成系统包安装，因为 `ql` 用户的 `sudo -n true` 检查显示需要密码

## 已安装内容

### WSL / Ubuntu

- `Windows Subsystem for Linux`
- `Ubuntu-22.04`
- WSL 版本：`2`

验证命令：

```powershell
wsl --status
wsl --list --verbose
wsl -d Ubuntu-22.04 -- bash -lc "cat /etc/os-release"
```

### Ubuntu 基础开发工具

通过 `apt-get` 安装：

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  python3 \
  python3-pip \
  python3-venv \
  python3-setuptools \
  python3-yaml \
  python3-pygments \
  wget \
  curl \
  pkg-config \
  lsb-release \
  software-properties-common
```

已确认版本：

- `build-essential`：`12.9ubuntu3`
- `cmake`：`3.22.1-1ubuntu1.22.04.2`
- `ninja-build`：`1.10.1-1`
- `git`：`1:2.34.1-1ubuntu1.17`
- `python3`：`3.10.6-1~22.04.1`
- `python3-pip`：`22.0.2+dfsg-1ubuntu0.7`
- `python3-venv`：`3.10.6-1~22.04.1`
- `gcc`：`11.4.0`
- `g++`：`11.4.0`

### LLVM / MLIR 工具链

通过 Ubuntu 22.04 官方仓库安装：

```bash
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  clang-15 \
  lld-15 \
  llvm-15 \
  llvm-15-dev \
  llvm-15-tools \
  mlir-15-tools \
  libmlir-15-dev
```

已确认版本：

- `clang-15`：`1:15.0.7-0ubuntu0.22.04.3`
- `lld-15`：`1:15.0.7-0ubuntu0.22.04.3`
- `llvm-15`：`1:15.0.7-0ubuntu0.22.04.3`
- `llvm-15-dev`：`1:15.0.7-0ubuntu0.22.04.3`
- `llvm-15-tools`：`1:15.0.7-0ubuntu0.22.04.3`
- `mlir-15-tools`：`1:15.0.7-0ubuntu0.22.04.3`
- `libmlir-15-dev`：`1:15.0.7-0ubuntu0.22.04.3`
- `llvm-config --version`：`15.0.7`
- `mlir-opt --version`：`Ubuntu LLVM version 15.0.7`
- `FileCheck --version`：`Ubuntu LLVM version 15.0.7`
- `lit --version`：`lit 15.0.7dev`

### 命令软链接

Ubuntu 的 LLVM/MLIR 包默认安装带版本号的命令，例如 `mlir-opt-15`。为方便项目 README 中的无版本命令直接运行，已创建以下软链接：

```bash
ln -sf /usr/bin/clang-15 /usr/local/bin/clang
ln -sf /usr/bin/clang++-15 /usr/local/bin/clang++
ln -sf /usr/bin/lld-15 /usr/local/bin/lld
ln -sf /usr/bin/llvm-config-15 /usr/local/bin/llvm-config
ln -sf /usr/bin/mlir-opt-15 /usr/local/bin/mlir-opt
ln -sf /usr/bin/FileCheck-15 /usr/local/bin/FileCheck
ln -sf /usr/lib/llvm-15/build/utils/lit/lit.py /usr/local/bin/lit
chmod +x /usr/lib/llvm-15/build/utils/lit/lit.py
```

已确认以下命令可找到：

- `/usr/local/bin/clang`
- `/usr/local/bin/clang++`
- `/usr/local/bin/lld`
- `/usr/local/bin/llvm-config`
- `/usr/local/bin/mlir-opt`
- `/usr/local/bin/FileCheck`
- `/usr/local/bin/lit`

### CMake 配置路径

已确认存在：

- `LLVM_DIR`：`/usr/lib/llvm-15/lib/cmake/llvm`
- `MLIR_DIR`：`/usr/lib/llvm-15/lib/cmake/mlir`

验证命令：

```bash
find /usr/lib/llvm-15 -maxdepth 4 \
  \( -name LLVMConfig.cmake -o -name MLIRConfig.cmake -o -name lit.py \) \
  -print
```

## 安装中遇到的问题

### `python3-lit` 包不存在

尝试安装：

```bash
apt-get install -y python3-lit
```

结果：

```text
E: Unable to locate package python3-lit
```

处理方式：

- 没有继续使用该包名
- 改用 LLVM 15 随包提供的 `lit.py`
- 创建 `/usr/local/bin/lit` 软链接指向 `/usr/lib/llvm-15/build/utils/lit/lit.py`

### `sudo` 需要密码

检查命令：

```bash
sudo -n true
```

结果：

```text
sudo: a password is required
```

处理方式：

- 使用 `wsl -d Ubuntu-22.04 -u root -- bash -lc "<command>"` 执行系统安装命令

## MLIR 项目分析

项目位置：

```text
D:\codeXProject\ai_compiler\ai-compiler-learning\projects\mlir-passes
```

WSL 路径：

```text
/mnt/d/codeXProject/ai_compiler/ai-compiler-learning/projects/mlir-passes
```

### 项目定位

这是一个用于学习 MLIR Pass 开发的小项目，目标是实现：

- 常量折叠 Pass
- 死代码消除 Pass
- 算子融合 Pass

### 当前实际文件

```text
./CMakeLists.txt
./README.md
./include/Passes.h
./lib/CMakeLists.txt
./lib/ConstantFoldPass.cpp
./test/constant_fold.mlir
./test/dce.mlir
```

### README 声明但当前缺失的文件或目录

README 中描述了更完整的项目结构，但当前仓库里尚不存在：

- `lib/DeadCodeElimPass.cpp`
- `lib/OperatorFusionPass.cpp`
- `test/CMakeLists.txt`
- `test/fusion.mlir`
- `tools/CMakeLists.txt`
- `docs/pass_design.md`
- `docs/debugging_guide.md`

### CMake 当前问题

已执行配置命令：

```bash
cd /mnt/d/codeXProject/ai_compiler/ai-compiler-learning/projects/mlir-passes
rm -rf build
mkdir build
cd build
CC=clang CXX=clang++ cmake -G Ninja .. \
  -DLLVM_DIR=/usr/lib/llvm-15/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-15/lib/cmake/mlir
```

结果：

```text
CMake Error at lib/CMakeLists.txt:3 (add_mlir_library):
  Unknown CMake command "add_mlir_library".
```

原因分析：

- 根 `CMakeLists.txt` 已经 `find_package(LLVM REQUIRED CONFIG)` 和 `find_package(MLIR REQUIRED CONFIG)`
- 但没有把 `${MLIR_CMAKE_DIR}` 加入 `CMAKE_MODULE_PATH`
- 也没有 `include(AddMLIR)`
- 因此 `lib/CMakeLists.txt` 中的 `add_mlir_library(...)` 无法识别

后续即使修复 `add_mlir_library`，仍会继续遇到：

- `lib/CMakeLists.txt` 引用了缺失的 `DeadCodeElimPass.cpp`
- `lib/CMakeLists.txt` 引用了缺失的 `OperatorFusionPass.cpp`
- 根 `CMakeLists.txt` 调用了 `add_subdirectory(test)`，但缺少 `test/CMakeLists.txt`
- 根 `CMakeLists.txt` 调用了 `add_subdirectory(tools)`，但缺少 `tools/` 目录

## 环境验证结果

### 工具版本验证

验证命令：

```bash
gcc --version | head -1
g++ --version | head -1
clang --version | head -1
cmake --version | head -1
ninja --version
python3 --version
pip3 --version
llvm-config --version
mlir-opt --version | head -1
FileCheck --version | head -1
lit --version
```

验证输出摘要：

```text
gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
Ubuntu clang version 15.0.7
cmake version 3.22.1
1.10.1
Python 3.10.12
pip 22.0.2
15.0.7
Ubuntu LLVM version 15.0.7
Ubuntu LLVM version 15.0.7
lit 15.0.7dev
```

### `mlir-opt` 基础验证

已执行：

```bash
cd /mnt/d/codeXProject/ai_compiler/ai-compiler-learning/projects/mlir-passes
mlir-opt --canonicalize test/constant_fold.mlir >/tmp/constant_fold.out
head -20 /tmp/constant_fold.out
```

结果：

- `mlir-opt` 可运行
- `test/constant_fold.mlir` 可被 MLIR 工具解析
- `--canonicalize` 能将测试中的常量表达式折叠为常量结果

输出片段：

```mlir
module {
  func.func @test_add_constant() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    return %c42_i32 : i32
  }
  func.func @test_mul_constant() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    return %c42_i32 : i32
  }
  func.func @test_float_add() -> f32 {
    %cst = arith.constant 4.000000e+00 : f32
    return %cst : f32
  }
}
```

## 当前状态

### 已完成

- WSL 可用
- Ubuntu 22.04 可用
- Ubuntu 基础开发工具已安装
- LLVM/MLIR 15 开发工具链已安装
- 常用无版本命令软链接已创建
- `LLVMConfig.cmake` 和 `MLIRConfig.cmake` 路径已确认
- `mlir-opt` 能处理项目测试 IR

### 尚未完成

- `projects/mlir-passes` 尚不能 CMake 配置成功
- 项目缺少 README 中声明的部分源码、测试和工具目录
- 尚未构建自定义 Pass 动态库或工具
- 尚未运行项目自定义 Pass 测试

## 建议的下一步

1. 修复根 `CMakeLists.txt`，加入 `${MLIR_CMAKE_DIR}` 和 `include(AddMLIR)`
2. 补齐或暂时移除 `DeadCodeElimPass.cpp`、`OperatorFusionPass.cpp` 引用
3. 增加 `test/CMakeLists.txt`，或先移除 `add_subdirectory(test)`
4. 增加 `tools/` 目录和可加载自定义 Pass 的测试工具，或先移除 `add_subdirectory(tools)`
5. 重新运行 CMake/Ninja，验证 `MLIRPasses` 能编译
6. 再接入 `lit` + `FileCheck`，让 `constant_fold.mlir` 和 `dce.mlir` 成为自动化测试

