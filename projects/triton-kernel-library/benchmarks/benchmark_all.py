#!/usr/bin/env python3
"""
全量性能基准测试

输出所有算子的性能对比数据和图表
"""

import torch
import sys
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '..')

from kernels.layernorm import layernorm
from kernels.gelu import gelu
from kernels.rmsnorm import rmsnorm
from kernels.flash_attn import flash_attention, standard_attention


def benchmark(func, *args, runs=100, warmup=10):
    """精确性能测试"""
    # 预热
    for _ in range(warmup):
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


def benchmark_layernorm():
    """LayerNorm 基准测试"""
    results = []
    
    shapes = [
        (1, 128, 512),
        (4, 256, 768),
        (8, 512, 1024),
        (16, 1024, 2048),
    ]
    
    for batch, seq, hidden in shapes:
        x = torch.randn((batch, seq, hidden), device='cuda')
        gamma = torch.ones(hidden, device='cuda')
        beta = torch.zeros(hidden, device='cuda')
        
        t_triton = benchmark(layernorm, x, gamma, beta, runs=50)
        t_pytorch = benchmark(torch.nn.functional.layer_norm, x, (hidden,), runs=50)
        
        results.append({
            'shape': (batch, seq, hidden),
            'triton_ms': t_triton,
            'pytorch_ms': t_pytorch,
            'speedup': t_pytorch / t_triton,
        })
    
    return results


def benchmark_gelu():
    """GELU 基准测试"""
    results = []
    
    shapes = [
        (1, 128, 512),
        (4, 256, 768),
        (8, 512, 1024),
        (16, 1024, 2048),
    ]
    
    for batch, seq, hidden in shapes:
        x = torch.randn((batch, seq, hidden), device='cuda')
        
        t_triton = benchmark(gelu, x, runs=50)
        t_pytorch = benchmark(torch.nn.functional.gelu, x, runs=50)
        
        results.append({
            'shape': (batch, seq, hidden),
            'triton_ms': t_triton,
            'pytorch_ms': t_pytorch,
            'speedup': t_pytorch / t_triton,
        })
    
    return results


def benchmark_rmsnorm():
    """RMSNorm 基准测试"""
    def rmsnorm_ref(x, gamma, eps=1e-6):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return gamma * x / rms
    
    results = []
    
    shapes = [
        (1, 128, 512),
        (4, 256, 768),
        (8, 512, 1024),
        (16, 1024, 2048),
    ]
    
    for batch, seq, hidden in shapes:
        x = torch.randn((batch, seq, hidden), device='cuda')
        gamma = torch.ones(hidden, device='cuda')
        
        t_triton = benchmark(rmsnorm, x, gamma, runs=50)
        t_pytorch = benchmark(rmsnorm_ref, x, gamma, runs=50)
        
        results.append({
            'shape': (batch, seq, hidden),
            'triton_ms': t_triton,
            'pytorch_ms': t_pytorch,
            'speedup': t_pytorch / t_triton,
        })
    
    return results


def benchmark_flash_attention():
    """FlashAttention 基准测试"""
    results = []
    
    shapes = [
        (2, 4, 256, 64),
        (4, 8, 512, 64),
        (4, 8, 1024, 64),
        (4, 8, 2048, 64),
    ]
    
    for batch, heads, seq, dim in shapes:
        q = torch.randn((batch, heads, seq, dim), device='cuda', dtype=torch.float16)
        k = torch.randn((batch, heads, seq, dim), device='cuda', dtype=torch.float16)
        v = torch.randn((batch, heads, seq, dim), device='cuda', dtype=torch.float16)
        
        try:
            t_triton = benchmark(flash_attention, q, k, v, runs=30)
            t_pytorch = benchmark(standard_attention, q, k, v, runs=30)
            
            results.append({
                'shape': (batch, heads, seq, dim),
                'triton_ms': t_triton,
                'pytorch_ms': t_pytorch,
                'speedup': t_pytorch / t_triton,
            })
        except Exception as e:
            print(f"  FlashAttention ({batch}, {heads}, {seq}, {dim}) 失败：{e}")
    
    return results


def print_results(name, results):
    """打印结果"""
    print(f"\n{name}")
    print("-" * 70)
    print(f"{'形状':<30} {'Triton(ms)':<12} {'PyTorch(ms)':<12} {'加速比':<10}")
    print("-" * 70)
    
    for r in results:
        shape_str = str(r['shape'])
        print(f"{shape_str:<30} {r['triton_ms']:>8.3f}   {r['pytorch_ms']:>8.3f}   {r['speedup']:>5.2f}x")
    
    # 平均加速比
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print("-" * 70)
    print(f"平均加速比：{avg_speedup:.2f}x")


def save_results(all_results):
    """保存结果到 JSON"""
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'benchmark_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存：{output_file}")
    
    # 同时更新最新结果
    latest_file = output_dir / 'latest.json'
    with open(latest_file, 'w') as f:
        json.dump(all_results, f, indent=2)


def main():
    """运行所有基准测试"""
    print("=" * 70)
    print("Triton Kernel Library - 全量性能基准测试")
    print("=" * 70)
    
    # 检查 GPU
    if not torch.cuda.is_available():
        print("错误：需要 NVIDIA GPU")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {gpu_name}")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行测试
    all_results = {
        'gpu': gpu_name,
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }
    
    print("\n" + "=" * 70)
    print("开始基准测试...")
    
    layernorm_results = benchmark_layernorm()
    all_results['results']['layernorm'] = layernorm_results
    print_results("LayerNorm", layernorm_results)
    
    gelu_results = benchmark_gelu()
    all_results['results']['gelu'] = gelu_results
    print_results("GELU", gelu_results)
    
    rmsnorm_results = benchmark_rmsnorm()
    all_results['results']['rmsnorm'] = rmsnorm_results
    print_results("RMSNorm", rmsnorm_results)
    
    flash_attn_results = benchmark_flash_attention()
    if flash_attn_results:
        all_results['results']['flash_attention'] = flash_attn_results
        print_results("FlashAttention", flash_attn_results)
    
    # 保存结果
    save_results(all_results)
    
    # 汇总
    print("\n" + "=" * 70)
    print("性能汇总")
    print("=" * 70)
    
    for kernel_name, results in all_results['results'].items():
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        print(f"  {kernel_name}: {avg_speedup:.2f}x")
    
    print("\n✓ 基准测试完成！")


if __name__ == "__main__":
    main()
