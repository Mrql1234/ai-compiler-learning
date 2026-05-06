// RUN: %mini_compiler_opt %s --mini-gpu-lowering | FileCheck %s
// CHECK: %[[OUT:.*]] = gpu.alloc  host_shared () : memref<2x8xf32>
// CHECK: %[[IN:.*]] = gpu.alloc  host_shared () : memref<2x4xf32>
// CHECK: memref.copy %arg0, %[[IN]]
// CHECK: %[[W:.*]] = gpu.alloc  host_shared () : memref<8x4xf32>
// CHECK: %[[B:.*]] = gpu.alloc  host_shared () : memref<8xf32>
// CHECK: gpu.launch_func
// CHECK: return %[[OUT]] : memref<2x8xf32>

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

  func.func @run() -> f32 attributes {llvm.emit_c_interface} {
    %input = arith.constant dense<[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]> : tensor<2x4xf32>
    %result = call @compute(%input) : (tensor<2x4xf32>) -> tensor<2x8xf32>
    %c0 = arith.constant 0 : index
    %first = tensor.extract %result[%c0, %c0] : tensor<2x8xf32>
    return %first : f32
  }
}
