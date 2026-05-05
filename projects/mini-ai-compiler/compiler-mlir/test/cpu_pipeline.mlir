// RUN: %mini_compiler_opt --mini-cpu-lowering %s | FileCheck %s

module {
  func.func @cpu_pipeline(%arg0: tensor<2x4xf32>) -> tensor<2x8xf32> {
    %w = "mini.constant"() {value = dense<0.0> : tensor<8x4xf32>} : () -> tensor<8x4xf32>
    %b = "mini.constant"() {value = dense<0.0> : tensor<8xf32>} : () -> tensor<8xf32>
    %0 = "mini.linear"(%arg0, %w, %b) : (tensor<2x4xf32>, tensor<8x4xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %1 = "mini.relu"(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }
}

// CHECK: llvm.func @malloc
// CHECK: llvm.func @cpu_pipeline
// CHECK: llvm.call @malloc
// CHECK: llvm.fadd
// CHECK: llvm.fmul
// CHECK-NOT: "mini.linear"
// CHECK-NOT: "mini.relu"
