// RUN: %mini_compiler_opt --pass-pipeline='builtin.module(func.func(mini-canonicalize,mini-fusion,mini-quantize-weights,mini-lower-to-linalg))' %s | FileCheck %s

module {
  func.func @qlinear_stays_late(%arg0: tensor<2x3xf32>) -> tensor<2x2xf32> {
    %weight = "mini.constant"() {value = dense<[[1.0, -2.0, 0.5], [0.25, 0.0, -0.25]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %bias = "mini.constant"() {value = dense<[0.5, -0.5]> : tensor<2xf32>} : () -> tensor<2xf32>
    %0 = "mini.linear"(%arg0, %weight, %bias) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

// CHECK-LABEL: func.func @qlinear_stays_late
// CHECK: arith.constant dense<{{.*}}> : tensor<2x3xi8>
// CHECK: "mini.qlinear"
// CHECK-NOT: arith.sitofp
