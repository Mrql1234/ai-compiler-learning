// RUN: %mini_compiler_opt --mini-quantized-gpu-prep %s | FileCheck %s

module {
  func.func @lower_quantized_linear(%arg0: tensor<2x3xf32>) -> tensor<2x2xf32> {
    %weight = "mini.constant"() {value = dense<[[1.0, -2.0, 0.5], [0.25, 0.0, -0.25]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %bias = "mini.constant"() {value = dense<[0.5, -0.5]> : tensor<2xf32>} : () -> tensor<2xf32>
    %0 = "mini.linear"(%arg0, %weight, %bias) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

// CHECK-LABEL: func.func @lower_quantized_linear
// CHECK: arith.constant dense<{{.*}}> : tensor<2x3xi8>
// CHECK: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: arith.sitofp
// CHECK: arith.mulf
// CHECK: arith.addf
