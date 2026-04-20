"""
TVM 矩阵乘法调度优化

优化技术：
1. 分块 (Tiling)
2. 并行化 (Parallelization)
3. 向量化 (Vectorization)
4. 内存重排 (Memory Reordering)
"""

import tvm
from tvm import te
import numpy as np
import time


def matmul_naive(M, N, K):
    """朴素矩阵乘法（无优化）"""
    A = te.placeholder((M, K), name='A')
    B = te.placeholder((K, N), name='B')
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    
    s = te.create_schedule(C.op)
    return s, [A, B, C]


def matmul_tiled(M, N, K, tile_size=32):
    """分块矩阵乘法"""
    A = te.placeholder((M, K), name='A')
    B = te.placeholder((K, N), name='B')
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    
    s = te.create_schedule(C.op)
    
    # 分块
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], tile_size, tile_size)
    
    return s, [A, B, C]


def matmul_optimized(M, N, K, tile_size=32):
    """优化版矩阵乘法（分块 + 并行 + 向量）"""
    A = te.placeholder((M, K), name='A')
    B = te.placeholder((K, N), name='B')
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    
    s = te.create_schedule(C.op)
    
    # 1. 分块
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], tile_size, tile_size)
    
    # 2. 并行化（最外层循环）
    s[C].parallel(xo)
    
    # 3. 向量化（最内层循环）
    s[C].vectorize(yi)
    
    # 4. 展开（适度展开减少循环开销）
    s[C].unroll(xi)
    
    return s, [A, B, C]


def matmul_with_reorder(M, N, K, tile_size=32):
    """带内存重排的矩阵乘法"""
    A = te.placeholder((M, K), name='A')
    B = te.placeholder((K, N), name='B')
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    
    s = te.create_schedule(C.op)
    
    # 分块
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], tile_size, tile_size)
    
    # 重排循环顺序
    ko, ki = s[C].split(k, factor=4)
    s[C].reorder(xo, yo, ko, xi, ki, yi)
    
    # 并行和向量
    s[C].parallel(xo)
    s[C].vectorize(yi)
    
    return s, [A, B, C]


def benchmark_schedule(schedule_func, M, N, K, target='llvm', runs=10):
    """测试不同调度方案的性能"""
    s, tensors = schedule_func(M, N, K)
    A, B, C = tensors
    
    # 编译
    f = tvm.build(s, [A, B, C], target=target)
    
    # 准备数据
    ctx = tvm.cpu(0)
    a = tvm.nd.array(np.random.rand(M, K).astype('float32'), ctx)
    b = tvm.nd.array(np.random.rand(K, N).astype('float32'), ctx)
    c = tvm.nd.array(np.zeros((M, N), dtype='float32'), ctx)
    
    # 预热
    f(a, b, c)
    
    # 测试
    start = time.time()
    for _ in range(runs):
        f(a, b, c)
    elapsed = time.time() - start
    
    return elapsed / runs * 1000  # ms


def compare_schedules(M, N, K):
    """对比不同调度方案"""
    print("=" * 70)
    print(f"矩阵乘法调度对比 (M={M}, N={N}, K={K})")
    print("=" * 70)
    
    schedules = [
        ("朴素版本", matmul_naive),
        ("分块优化", matmul_tiled),
        ("分块 + 并行 + 向量", matmul_optimized),
        ("完整优化", matmul_with_reorder),
    ]
    
    results = []
    baseline = None
    
    for name, func in schedules:
        try:
            time_ms = benchmark_schedule(func, M, N, K)
            results.append((name, time_ms))
            
            if baseline is None:
                baseline = time_ms
            
            speedup = baseline / time_ms
            print(f"{name:<25} {time_ms:>8.3f} ms  ({speedup:>5.2f}x)")
        except Exception as e:
            print(f"{name:<25} 错误：{e}")
    
    return results


if __name__ == "__main__":
    # 测试配置
    sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]
    
    for M, N, K in sizes:
        print()
        compare_schedules(M, N, K)
    
    print("\n✓ 测试完成！")
