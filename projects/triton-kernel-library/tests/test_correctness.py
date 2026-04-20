#!/usr/bin/env python3
"""
正确性测试 - 对比 Triton 实现与 PyTorch 参考实现

所有算子的数值差异应 < 1e-4
"""

import torch
import sys
sys.path.insert(0, '..')

from kernels.layernorm import layernorm
from kernels.gelu import gelu
from kernels.rmsnorm import rmsnorm
from kernels.flash_attn import flash_attention, standard_attention


def test_layernorm():
    """测试 LayerNorm"""
    print("测试 LayerNorm...")
    
    shapes = [
        (1, 128, 512),
        (4, 256, 768),
        (8, 512, 1024),
    ]
    
    all_passed = True
    for batch, seq, hidden in shapes:
        x = torch.randn((batch, seq, hidden), device='cuda')
        gamma = torch.ones(hidden, device='cuda')
        beta = torch.zeros(hidden, device='cuda')
        
        y_triton = layernorm(x, gamma, beta)
        y_ref = torch.nn.functional.layer_norm(x, (hidden,))
        
        diff = torch.max(torch.abs(y_triton - y_ref))
        passed = diff < 1e-4
        
        status = "✓" if passed else "✗"
        print(f"  {status} ({batch}, {seq}, {hidden}): max_diff={diff:.6f}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_gelu():
    """测试 GELU"""
    print("测试 GELU...")
    
    shapes = [
        (1, 128, 512),
        (4, 256, 768),
        (8, 512, 1024),
    ]
    
    all_passed = True
    for batch, seq, hidden in shapes:
        x = torch.randn((batch, seq, hidden), device='cuda')
        
        y_triton = gelu(x)
        y_ref = torch.nn.functional.gelu(x)
        
        diff = torch.max(torch.abs(y_triton - y_ref))
        passed = diff < 1e-4
        
        status = "✓" if passed else "✗"
        print(f"  {status} ({batch}, {seq}, {hidden}): max_diff={diff:.6f}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_rmsnorm():
    """测试 RMSNorm"""
    print("测试 RMSNorm...")
    
    def rmsnorm_ref(x, gamma, eps=1e-6):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return gamma * x / rms
    
    shapes = [
        (1, 128, 512),
        (4, 256, 768),
        (8, 512, 1024),
    ]
    
    all_passed = True
    for batch, seq, hidden in shapes:
        x = torch.randn((batch, seq, hidden), device='cuda')
        gamma = torch.ones(hidden, device='cuda')
        
        y_triton = rmsnorm(x, gamma)
        y_ref = rmsnorm_ref(x, gamma)
        
        diff = torch.max(torch.abs(y_triton - y_ref))
        passed = diff < 1e-4
        
        status = "✓" if passed else "✗"
        print(f"  {status} ({batch}, {seq}, {hidden}): max_diff={diff:.6f}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_flash_attention():
    """测试 FlashAttention"""
    print("测试 FlashAttention...")
    
    shapes = [
        (2, 4, 256, 64),
        (4, 8, 512, 64),
    ]
    
    all_passed = True
    for batch, heads, seq, dim in shapes:
        q = torch.randn((batch, heads, seq, dim), device='cuda', dtype=torch.float16)
        k = torch.randn((batch, heads, seq, dim), device='cuda', dtype=torch.float16)
        v = torch.randn((batch, heads, seq, dim), device='cuda', dtype=torch.float16)
        
        try:
            o_triton = flash_attention(q, k, v, causal=False)
            o_ref = standard_attention(q, k, v, causal=False)
            
            diff = torch.max(torch.abs(o_triton - o_ref))
            passed = diff < 1e-2  # FP16 精度要求较低
            
            status = "✓" if passed else "✗"
            print(f"  {status} ({batch}, {heads}, {seq}, {dim}): max_diff={diff:.4f}")
            
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  ✗ ({batch}, {heads}, {seq}, {dim}): {e}")
            all_passed = False
    
    return all_passed


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Triton Kernel Library - 正确性测试")
    print("=" * 60)
    print()
    
    # 检查 GPU
    if not torch.cuda.is_available():
        print("错误：需要 NVIDIA GPU")
        return False
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 运行测试
    results = []
    results.append(("LayerNorm", test_layernorm()))
    results.append(("GELU", test_gelu()))
    results.append(("RMSNorm", test_rmsnorm()))
    results.append(("FlashAttention", test_flash_attention()))
    
    # 汇总
    print()
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ 所有测试通过！")
        return True
    else:
        print("✗ 部分测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
