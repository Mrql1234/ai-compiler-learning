// RUN: %mini_compiler_opt %s --pass-pipeline='builtin.module(func.func(mini-canonicalize,mini-fusion),mini-gpu-runtime-call-lowering{backend=cuda_hand})' | FileCheck %s --check-prefix=CUDA
// RUN: %mini_compiler_opt %s --pass-pipeline='builtin.module(func.func(mini-canonicalize,mini-fusion),mini-gpu-runtime-call-lowering{backend=cublas})' | FileCheck %s --check-prefix=CUBLAS
// RUN: %mini_compiler_opt %s --pass-pipeline='builtin.module(mini-gpu-runtime-call-lowering-pipeline{backend=cuda_hand})' | FileCheck %s --check-prefix=LLVM-CUDA

module {
  func.func @compute(%arg0: tensor<2x4xf32>) -> tensor<2x8xf32> {
    %w = "mini.constant"() {value = dense<[
      [1.0, 2.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0]
    ]> : tensor<8x4xf32>} : () -> tensor<8x4xf32>
    %b = "mini.constant"() {value = dense<[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]> : tensor<8xf32>} : () -> tensor<8xf32>
    %0 = "mini.linear"(%arg0, %w, %b) : (tensor<2x4xf32>, tensor<8x4xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %1 = "mini.relu"(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }
}

// CUDA: func.func private @mini_cuda_linear_relu_f32_memref
// CUDA-LABEL: func.func @compute
// CUDA: bufferization.to_buffer {{.*}} : tensor<2x4xf32> to memref<2x4xf32>
// CUDA: memref.alloc() : memref<2x8xf32>
// CUDA: call @mini_cuda_linear_relu_f32_memref
// CUDA-SAME: memref<2x4xf32>, memref<8x4xf32>, memref<8xf32>, memref<2x8xf32>, index, index, index
// CUDA: bufferization.to_tensor {{.*}} restrict writable : memref<2x8xf32> to tensor<2x8xf32>

// CUBLAS: func.func private @mini_cublas_linear_relu_f32_memref
// CUBLAS-LABEL: func.func @compute
// CUBLAS: call @mini_cublas_linear_relu_f32_memref
// CUBLAS-SAME: memref<2x4xf32>, memref<8x4xf32>, memref<8xf32>, memref<2x8xf32>, index, index, index

// LLVM-CUDA: llvm.func @mini_cuda_linear_relu_f32_memref
// LLVM-CUDA: llvm.func @compute
// LLVM-CUDA: llvm.call @mini_cuda_linear_relu_f32_memref
