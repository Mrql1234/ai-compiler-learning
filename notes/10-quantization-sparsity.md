# 08 - 量化与稀疏化优化：模型压缩与加速

> 📅 学习日期：2026-04-15  
> 📚 阶段：阶段 3 - LLM 推理优化  
> ⏱️ 预计耗时：2-3 周  
> 💻 平台：Linux/Windows + NVIDIA GPU（支持 Tensor Core）  
> 🔗 前置：[06-GEMM 优化实战](./06-gemm-optimization.md), [08-算子融合](./08-operator-fusion.md)

---

## 🎯 学习目标

学完这篇，你应该能：

1. 理解 **量化的基本原理和收益**
2. 掌握 **PTQ（训练后量化）** 和 **QAT（量化感知训练）** 的区别
3. 理解 **稀疏化加速** 的原理
4. 能用 **TensorRT / TVM** 部署量化模型
5. 理解 **Tensor Core** 对量化的硬件支持

---

## 💡 为什么需要量化？

### 量化的核心价值

**定义**：量化 = 用**低精度**数据类型表示模型权重和激活值。

```
FP32 (32 位浮点) → FP16 (16 位浮点) → INT8 (8 位整数) → INT4 (4 位整数)
     ↓                ↓                  ↓                 ↓
   100%             50% 大小           25% 大小          12.5% 大小
   1x 带宽         2x 带宽节省        4x 带宽节省       8x 带宽节省
```

**收益**：
1. **模型大小减小** - INT8 是 FP32 的 1/4
2. **内存带宽降低** - 减少 4x 内存访问
3. **计算加速** - INT8 Tensor Core 可达 FP32 的 8-16x
4. **功耗降低** - 低精度计算能耗更低

### 📊 量化性能对比（A100 GPU）

| 精度 | 模型大小 | 内存带宽 | 计算吞吐 | 相对 FP32 |
|------|---------|---------|---------|----------|
| FP32 | 100% | 1.5 TB/s | 19.5 TFLOPS | 1x |
| FP16 | 50% | 1.5 TB/s | 156 TFLOPS* | 8x |
| INT8 | 25% | 1.5 TB/s | 312 TFLOPS* | 16x |
| FP8 | 12.5% | 1.5 TB/s | 312 TFLOPS* | 16x |

\* 使用 Tensor Core

**关键洞察**：
- Tensor Core 是量化的**硬件基础**
- INT8/FP16 的**计算吞吐**远高于 FP32
- 但量化可能带来**精度损失**

---

## 📚 量化基础概念

### 1. 量化参数

**核心公式**：

```
量化：q = round(r / scale) + zero_point
反量化：r = (q - zero_point) * scale

其中：
  r = 实数值（FP32）
  q = 量化值（INT8）
  scale = 缩放因子（浮点数）
  zero_point = 零点偏移（整数，通常为 0 或 128）
```

**示例**：

```python
# FP32 → INT8 量化
def quantize_fp32_to_int8(data, scale, zero_point=128):
    # 量化
    q = np.round(data / scale) + zero_point
    # 截断到 INT8 范围
    q = np.clip(q, 0, 255).astype(np.uint8)
    return q

# INT8 → FP32 反量化
def dequantize_int8_to_fp32(q, scale, zero_point=128):
    return (q.astype(np.float32) - zero_point) * scale

# 示例：量化一个权重矩阵
weight_fp32 = np.random.randn(128, 128).astype(np.float32)
scale = np.max(np.abs(weight_fp32)) / 127.0  # 对称量化
weight_int8 = quantize_fp32_to_int8(weight_fp32, scale, zero_point=0)

# 验证
weight_dequant = dequantize_int8_to_fp32(weight_int8, scale, 0)
error = np.max(np.abs(weight_fp32 - weight_dequant))
print(f"最大量化误差：{error:.6f}")
```

### 2. 量化类型

#### 对称量化 vs 非对称量化

```
对称量化：
  zero_point = 0
  量化范围：[-127, 127] (INT8)
  适合：权重（通常分布对称）

非对称量化：
  zero_point ≠ 0 (通常 128)
  量化范围：[0, 255] (UINT8)
  适合：激活值（ReLU 后非负）
```

#### 逐层量化 vs 逐通道量化

```
逐层量化 (Per-Tensor)：
  - 整个 Tensor 用一个 scale
  - 简单，但精度损失大
  
逐通道量化 (Per-Channel)：
  - 每个输出通道用不同的 scale
  - 精度更高，推荐用于卷积权重

示例：Conv2d 权重 (out_channels, in_channels, kH, kW)
  - 逐层：1 个 scale
  - 逐通道：out_channels 个 scale（每个输出通道一个）
```

### 3. 量化方式

| 方式 | 描述 | 精度 | 难度 |
|------|------|------|------|
| **动态量化** | 权重静态量化，激活值动态量化 | 中 | 低 |
| **静态量化** | 权重和激活值都静态量化 | 高 | 中 |
| **量化感知训练 (QAT)** | 训练时模拟量化 | 最高 | 高 |

---

## 🔥 实战 1：训练后量化（PTQ）

### PTQ 流程

```
1. 准备校准数据集（100-1000 个样本）
2. 用 FP32 模型跑校准数据，收集激活值分布
3. 计算每层的 scale 和 zero_point
4. 量化权重
5. 验证精度
```

### PyTorch PTQ 示例

```python
# ptq_example.py
import torch
import torch.quantization as quant
from torchvision import models

# 1. 加载预训练模型
model = models.resnet18(pretrained=True)
model.eval()

# 2. 准备校准数据集
def prepare_calibration_data(num_samples=100):
    """准备校准数据"""
    calibration_data = []
    for i in range(num_samples):
        # 随机输入（实际应该用真实数据）
        x = torch.randn(1, 3, 224, 224)
        calibration_data.append(x)
    return calibration_data

calibration_data = prepare_calibration_data(100)

# 3. 配置量化
model.qconfig = quant.get_default_qconfig('fbgemm')  # CPU 后端
# model.qconfig = quant.get_default_qconfig('fbgemm')  # GPU 后端用 'fbgemm'

# 4. 准备量化（插入观察器）
model_prepared = quant.prepare(model)

# 5. 校准（跑数据，收集激活值分布）
print("校准中...")
with torch.no_grad():
    for data in calibration_data:
        model_prepared(data)

# 6. 转换（应用量化）
model_quantized = quant.convert(model_prepared)

# 7. 验证精度
def benchmark(model, input, runs=100):
    import time
    model.eval()
    
    # 预热
    with torch.no_grad():
        model(input)
    
    # 测试
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            model(input)
    end = time.time()
    
    return (end - start) / runs * 1000

# 性能对比
input_tensor = torch.randn(1, 3, 224, 224)

print("\n性能对比（ResNet-18, CPU）：")
fp32_time = benchmark(model, input_tensor)
print(f"FP32: {fp32_time:.2f} ms")

int8_time = benchmark(model_quantized, input_tensor)
print(f"INT8: {int8_time:.2f} ms")

print(f"加速比：{fp32_time / int8_time:.2f}x")

# 精度验证
with torch.no_grad():
    fp32_output = model(input_tensor)
    int8_output = model_quantized(input_tensor)
    
    diff = torch.max(torch.abs(fp32_output - int8_output))
    print(f"\n输出差异：{diff:.6f}")
```

**预期输出**（CPU）：

```
校准中...

性能对比（ResNet-18, CPU）：
FP32: 45.32 ms
INT8: 18.45 ms
加速比：2.46x

输出差异：0.023456
```

---

## 🔥 实战 2：量化感知训练（QAT）

### QAT 流程

```
1. 从预训练 FP32 模型开始
2. 插入"伪量化"节点（模拟量化误差）
3. 微调训练（1-10 epochs）
4. 导出量化模型
```

### PyTorch QAT 示例

```python
# qat_example.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quant
from torchvision import models, datasets, transforms

# 1. 加载预训练模型
model = models.resnet18(pretrained=True)
model.eval()

# 2. 配置 QAT
model.qconfig = quant.get_default_qat_qconfig('fbgemm')

# 3. 准备 QAT（插入伪量化节点）
model_prepared = quant.prepare_qat(model, inplace=False)

# 4. 微调训练
# 准备数据
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('/path/to/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 微调（实际训练中应该更多 epoch）
model_prepared.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_prepared.parameters(), lr=0.001)

print("QAT 微调中...")
for epoch in range(3):  # 简化：只训练 3 个 epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model_prepared(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

# 5. 转换为量化模型
model_quantized = quant.convert(model_prepared, inplace=False)

# 6. 验证
model_quantized.eval()
print("\nQAT 量化完成！")
print(f"模型大小：{sum(p.numel() for p in model_quantized.parameters())} 参数")

# 保存量化模型
torch.save(model_quantized.state_dict(), 'resnet18_int8_qat.pth')
```

### PTQ vs QAT 对比

| 特性 | PTQ | QAT |
|------|-----|-----|
| 训练数据 | 只需校准数据 | 需要训练数据 |
| 训练时间 | 无 | 1-10 epochs |
| 精度损失 | 较大（1-3%） | 较小（<1%） |
| 适用场景 | 快速部署 | 精度敏感场景 |
| 实现难度 | 低 | 中 |

---

## 🔥 实战 3：Tensor Core 编程

### Tensor Core 基础

**什么是 Tensor Core？**
- NVIDIA Volta (V100) 及以后架构引入的**专用矩阵计算单元**
- 专门用于**矩阵乘法累加**（GEMM）
- 支持 FP16、INT8、TF32、FP8 等低精度

**Tensor Core 操作**：

```
D = A × B + C

其中：
  A: M×K 矩阵（FP16/INT8）
  B: K×N 矩阵（FP16/INT8）
  C: M×N 矩阵（FP32/FP16）累加器
  D: M×N 矩阵（FP32/FP16）输出

A100 Tensor Core 吞吐：
  - FP16: 312 TFLOPS
  - INT8: 624 TOPS
  - FP8: 312 TFLOPS (H100)
```

### CUDA WMMA API 示例

```cuda
// tensor_core_gemm.cu
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

// Tensor Core 要求：
// - M, N, K 必须是 16 的倍数
// - 内存对齐：16 bytes

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void tensorCoreGemm(half *A, half *B, float *C, int M, int K, int N) {
    // 计算线程负责的输出块位置
    int row = blockIdx.y * WMMA_M;
    int col = blockIdx.x * WMMA_N;
    
    // 声明 WMMA 矩阵片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // 初始化累加器
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 分块循环
    for (int i = 0; i < K; i += WMMA_K) {
        // 加载 A 块
        if (row < M && i < K) {
            wmma::load_matrix_sync(a_frag, A + row * K + i, K);
        }
        
        // 加载 B 块
        if (i < K && col < N) {
            wmma::load_matrix_sync(b_frag, B + i * N + col, N);
        }
        
        // Tensor Core 矩阵乘法
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 存储结果
    if (row < M && col < N) {
        wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
    }
}

// 性能测试
void benchmark_tensor_core(int M, int K, int N) {
    // 分配内存（省略...）
    
    // 启动配置
    dim3 blockDim(256);
    dim3 gridDim((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    
    // 性能测试
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热
    tensorCoreGemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    
    // 正式测试
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        tensorCoreGemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    elapsed /= 100;
    
    // 计算性能
    float tflops = (2.0f * M * K * N) / (elapsed * 1e6) / 1000;
    printf("Tensor Core GEMM %dx%dx%d: %.2f ms, %.1f TFLOPS\n", M, K, N, elapsed, tflops);
}

int main() {
    // 检查 Tensor Core 支持
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Tensor Core 支持：%s\n", (prop.major >= 7) ? "是" : "否");
    
    if (prop.major >= 7) {
        benchmark_tensor_core(4096, 4096, 4096);
    } else {
        printf("此 GPU 不支持 Tensor Core\n");
    }
    
    return 0;
}
```

### 编译和运行

```bash
# 编译（需要 Compute Capability >= 7.0）
nvcc -O3 -arch=sm_70 tensor_core_gemm.cu -o tensor_core_benchmark

# 运行
./tensor_core_benchmark
```

**预期输出**（A100）：

```
GPU: NVIDIA A100-SXM4-40GB
Compute Capability: 8.0
Tensor Core 支持：是
Tensor Core GEMM 4096x4096x4096: 0.045 ms, 305.2 TFLOPS
```

---

## 🔥 实战 4：LLM 量化（AWQ/GPTQ）

### LLM 量化挑战

**问题**：LLM 模型**权重分布不均匀**，直接量化精度损失大。

```
LLaMA-7B 权重量化对比：
  - 朴素 INT8： perplexity 上升 5-10%
  - AWQ (Activation-aware)：perplexity 上升 <1%
  - GPTQ：perplexity 上升 <1%
```

### AWQ 核心思想

**Activation-aware Weight Quantization**：
- 用**激活值幅度**指导权重量化
- 保护**重要权重**（对应大激活值）
- 用**缩放因子**调整权重分布

```python
# AWQ 简化示例
import torch

def awq_quantize(weight, activation, bits=4):
    """
    AWQ 量化：根据激活值保护重要权重
    """
    # 1. 计算激活值幅度（重要性指标）
    act_scale = activation.abs().mean(dim=0)
    
    # 2. 根据重要性缩放权重
    # 重要权重（大激活）用更小的 scale，保留更多精度
    weight_scale = act_scale.pow(0.5)  # 可调整指数
    
    # 3. 量化
    max_val = 2 ** (bits - 1) - 1
    scale = weight.abs().max() / max_val
    
    # 4. 应用缩放
    weight_scaled = weight * weight_scale
    q_weight = torch.round(weight_scaled / scale).clamp(-max_val, max_val)
    
    # 5. 保存参数
    return q_weight, scale, weight_scale

# 使用示例
weight = torch.randn(4096, 4096)  # LLM 权重
activation = torch.randn(1, 4096)  # 校准激活值

q_weight, scale, weight_scale = awq_quantize(weight, activation, bits=4)
print(f"4-bit 量化完成，保存参数：scale={scale:.6f}")
```

### 用 AutoAWQ 量化 LLM

```python
# autoawq_example.py
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 1. 加载模型
model_path = "mistralai/Mistral-7B-v0.1"
print(f"加载模型：{model_path}")

model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    use_cache=False
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. 量化
print("量化中（可能需要 10-30 分钟）...")
quant_config = {
    "zero_point": True,      # 使用零点
    "q_group_size": 128,     # 分组大小
    "w_bit": 4,              # 4-bit 量化
    "version": "GEMM"        # 使用 GEMM 格式
}

model.quantize(tokenizer, quant_config=quant_config)

# 3. 保存
model.save_quantized("./mistral-7b-awq-4bit")
tokenizer.save_pretrained("./mistral-7b-awq-4bit")

print("✅ 量化完成！")
print(f"量化后模型大小：{sum(p.numel() for p in model.model.parameters()) * 4 / 1e9:.2f} GB")

# 4. 加载并测试
print("\n加载量化模型...")
model = AutoAWQForCausalLM.from_quantized("./mistral-7b-awq-4bit")

# 推理测试
prompt = "Hello, I'm a language model,"
inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7
)

print(f"\n生成结果：{tokenizer.decode(output[0])}")
```

### 量化效果对比

| 模型 | 精度 | 大小 | 推理速度 | Perplexity 变化 |
|------|------|------|---------|----------------|
| LLaMA-7B | FP16 | 14 GB | 1x | - |
| LLaMA-7B | INT8 | 7 GB | 1.8x | +0.5% |
| LLaMA-7B | AWQ 4-bit | 3.5 GB | 2.5x | +1.2% |
| LLaMA-7B | GPTQ 4-bit | 3.5 GB | 2.5x | +1.0% |

---

## 📊 稀疏化优化

### 结构化稀疏

**定义**：以**固定模式**剪枝权重，硬件可加速。

```
NVIDIA Ampere (A100) 支持 2:4 结构化稀疏：
  - 每 4 个权重中最多 2 个非零
  - 硬件跳过零值计算
  - 理论 2x 加速

示例：
原始：[w1, w2, w3, w4, w5, w6, w7, w8]
2:4 稀疏：[w1, 0, w3, 0, 0, w6, 0, w8]  ← 每 4 个最多 2 个非零
```

### PyTorch 稀疏化示例

```python
# sparsity_example.py
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn

# 定义模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)

model = SimpleNet()

# 打印原始参数
print("原始模型：")
print(f"  linear1 权重形状：{model.linear1.weight.shape}")
print(f"  非零元素：{torch.count_nonzero(model.linear1.weight)} / {model.linear1.weight.numel()}")

# 应用 L1 非结构化剪枝
print("\n应用 L1 非结构化剪枝（50%）...")
prune.l1_unstructured(model.linear1, name='weight', amount=0.5)

print(f"  非零元素：{torch.count_nonzero(model.linear1.weight)} / {model.linear1.weight.numel()}")

# 应用结构化剪枝（2:4）
print("\n应用 2:4 结构化剪枝...")
prune.ln_structured(model.linear2, name='weight', amount=0.5, n=4, dim=0)

print(f"  linear2 非零元素：{torch.count_nonzero(model.linear2.weight)} / {model.linear2.weight.numel()}")

# 永久移除剪枝
prune.remove(model.linear1, 'weight')
prune.remove(model.linear2, 'weight')

# 性能测试
def benchmark(model, input, runs=1000):
    import time
    model.eval()
    
    # 预热
    with torch.no_grad():
        model(input)
    
    # 测试
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            model(input)
    end = time.time()
    
    return (end - start) / runs * 1000

input_tensor = torch.randn(32, 512)
print(f"\n推理延迟：{benchmark(model, input_tensor):.3f} ms")
```

### 稀疏化加速对比

| 稀疏类型 | 稀疏度 | 理论加速 | 实际加速 (A100) |
|---------|-------|---------|----------------|
| 非结构化 | 50% | 2x | 1.1x* |
| 2:4 结构化 | 50% | 2x | 1.8x |
| 4:8 结构化 | 50% | 2x | 1.7x |

\* 非结构化稀疏需要专用硬件支持，否则加速有限

---

## ✅ 本周任务清单

### 必做（核心）

- [ ] 用 PyTorch 实现 PTQ（训练后量化）
- [ ] 对比 FP32 和 INT8 的性能/精度
- [ ] 理解 Tensor Core 的工作原理
- [ ] 在笔记里记录量化实验结果

### 选做（深入）

- [ ] 尝试 QAT（量化感知训练）
- [ ] 用 AutoAWQ 量化一个 LLM 模型
- [ ] 实现简单的 2:4 结构化稀疏
- [ ] 阅读 AWQ/GPTQ 论文

### 挑战任务

- [ ] 量化一个 7B+ 的 LLM 模型
- [ ] 对比不同量化方法（PTQ/QAT/AWQ/GPTQ）的精度
- [ ] 部署量化模型到生产环境

---

## 📚 参考资料

- **量化库**：
  - PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
  - TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/
  - AutoAWQ: https://github.com/casper-hansen/AutoAWQ
  - GPTQ: https://github.com/IST-DASLab/gptq

- **论文**：
  - AWQ: Activation-aware Weight Quantization (2023)
  - GPTQ: Accurate Post-Training Quantization for GPT (2022)
  - Tensor Cores: https://docs.nvidia.com/deeplearning/performance/mixed-precision/

- **教程**：
  - NVIDIA Tensor Core 编程：https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
  - PyTorch 量化教程：https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html

---

## 🔗 下一篇预告

**笔记 09**：LLM 推理优化 - KV Cache 与 Continuous Batching

- LLM 推理的特殊挑战
- KV Cache 原理和实现
- PagedAttention（vLLM 核心技术）
- Continuous Batching 调度

---

_笔记创建：2026-04-15_  
_适合人群：有 CUDA 基础，想深入 LLM 推理优化_  
_平台：Linux/Windows + NVIDIA GPU（推荐 A100/H100/RTX 4090）_  
_难度：⭐⭐⭐⭐（需要理解量化理论和 Tensor Core）_
