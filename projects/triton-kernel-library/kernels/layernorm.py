"""
Triton LayerNorm 实现

性能：1.39x vs PyTorch
优化点：
- 每个 program 处理一个样本（行）
- 两次遍历：mean/var → normalize
- 融合 gamma/beta 缩放
"""

import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    # 输入指针
    x_ptr,
    # 权重指针
    w_ptr,
    b_ptr,
    # 输出指针
    y_ptr,
    # 中间结果指针（用于反向传播）
    mean_ptr,
    rstd_ptr,
    # 步长
    stride,
    # 编译时常量
    N: tl.constexpr,  # hidden dimension
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    LayerNorm 前向传播 kernel
    
    每个 program 处理一个样本（一行）
    
    参数:
        x_ptr: 输入张量指针 (batch * seq_len, hidden)
        w_ptr: gamma 权重指针 (hidden,)
        b_ptr: beta 权重指针 (hidden,)
        y_ptr: 输出张量指针
        mean_ptr: 保存 mean (用于反向传播)
        rstd_ptr: 保存 1/std (用于反向传播)
        stride: 行步长
        N: hidden dimension
        eps: 数值稳定性 eps
        BLOCK_SIZE: block 大小（2 的幂次）
    """
    # 1. 计算 program ID（样本 ID）
    row = tl.program_id(0)
    
    # 2. 计算这一行的指针偏移
    row_start = row * stride
    y_ptr_row = y_ptr + row_start
    x_ptr_row = x_ptr + row_start
    
    # 3. 创建列索引
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # 4. 加载输入数据
    x = tl.load(x_ptr_row + cols, mask=mask, other=0.0)
    
    # 5. 计算 mean
    mean = tl.sum(x, axis=0) / N
    
    # 6. 计算 variance
    x_mean = x - mean
    var = tl.sum(x_mean * x_mean, axis=0) / N
    
    # 7. 计算 1/std
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # 8. 保存中间结果（用于反向传播）
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)
    
    # 9. 归一化
    x_norm = x_mean * rstd
    
    # 10. 应用 gamma 和 beta
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    y = w * x_norm + b
    
    # 11. 存储结果
    tl.store(y_ptr_row + cols, y, mask=mask)


class LayerNormKernel:
    """LayerNorm 封装类"""
    
    @staticmethod
    def forward(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        """
        LayerNorm 前向传播
        
        参数:
            x: 输入张量 (batch, seq_len, hidden)
            gamma: 权重 (hidden,)
            beta: 偏置 (hidden,)
            eps: 数值稳定性
            
        返回:
            y: 输出张量 (batch, seq_len, hidden)
        """
        # 保存原始形状
        batch, seq_len, hidden = x.shape
        
        # 展平为 2D (batch * seq_len, hidden)
        x_flat = x.view(-1, hidden).contiguous()
        
        # 创建输出张量
        y_flat = torch.empty_like(x_flat)
        
        # 创建中间结果张量
        rows = batch * seq_len
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        
        # 配置 BLOCK_SIZE（2 的幂次）
        BLOCK_SIZE = triton.next_power_of_2(hidden)
        
        # 配置 grid
        grid = (rows,)
        
        # 启动 kernel
        layernorm_kernel[grid](
            x_flat, gamma, beta, y_flat, mean, rstd,
            x_flat.stride(0),
            hidden,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # 恢复原始形状
        y = y_flat.view(batch, seq_len, hidden)
        
        return y, mean, rstd


def layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
    """
    LayerNorm 便捷函数
    
    参数:
        x: 输入张量 (batch, seq_len, hidden)
        gamma: 权重 (hidden,)
        beta: 偏置 (hidden,)
        eps: 数值稳定性
        
    返回:
        y: 输出张量 (batch, seq_len, hidden)
    """
    y, _, _ = LayerNormKernel.forward(x, gamma, beta, eps)
    return y


# ===== 测试和 Benchmark =====

if __name__ == "__main__":
    import time
    
    def benchmark(func, *args, runs=100):
        """性能测试"""
        # 预热
        func(*args)
        torch.cuda.synchronize()
        
        # 使用 Event 精确计时
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(runs):
            func(*args)
        end.record()
        torch.cuda.synchronize()
        
        return start.elapsed_time(end) / runs
    
    # 测试配置
    batch, seq_len, hidden = 32, 512, 768
    device = 'cuda'
    
    print("=" * 60)
    print(f"Triton LayerNorm 测试 (batch={batch}, seq={seq_len}, hidden={hidden})")
    print("=" * 60)
    
    # 准备数据
    x = torch.randn((batch, seq_len, hidden), device=device)
    gamma = torch.ones(hidden, device=device)
    beta = torch.zeros(hidden, device=device)
    
    # 1. 正确性测试
    print("\n1. 正确性测试...")
    y_triton = layernorm(x, gamma, beta)
    y_ref = torch.nn.functional.layer_norm(x, (hidden,))
    
    diff = torch.max(torch.abs(y_triton - y_ref))
    print(f"   最大差异：{diff:.6f}")
    print(f"   结果正确：{'✓' if diff < 1e-4 else '✗'}")
    
    # 2. 性能测试
    print("\n2. 性能测试...")
    triton_time = benchmark(layernorm, x, gamma, beta)
    pytorch_time = benchmark(torch.nn.functional.layer_norm, x, (hidden,))
    
    speedup = pytorch_time / triton_time
    print(f"   Triton:   {triton_time:.3f} ms")
    print(f"   PyTorch:  {pytorch_time:.3f} ms")
    print(f"   加速比：  {speedup:.2f}x")
    
    # 3. 不同形状测试
    print("\n3. 不同形状性能对比...")
    print(f"   {'形状':<20} {'Triton(ms)':<12} {'PyTorch(ms)':<12} {'加速比':<8}")
    print(f"   {'-'*52}")
    
    shapes = [
        (1, 128, 512),
        (4, 256, 768),
        (8, 512, 1024),
        (16, 1024, 2048),
    ]
    
    for b, s, h in shapes:
        x_test = torch.randn((b, s, h), device=device)
        gamma_test = torch.ones(h, device=device)
        beta_test = torch.zeros(h, device=device)
        
        t_triton = benchmark(layernorm, x_test, gamma_test, beta_test, runs=50)
        t_pytorch = benchmark(torch.nn.functional.layer_norm, x_test, (h,), runs=50)
        speedup = t_pytorch / t_triton
        
        print(f"   ({b:>2}, {s:>4}, {h:>4})  {t_triton:>8.3f}   {t_pytorch:>8.3f}   {speedup:>5.2f}x")
    
    print("\n✓ 测试完成！")
