// RUN: %mini_compiler_opt --mini-quantize-weights %s | FileCheck %s

module {
  func.func @quantize_linear_weight(%arg0: tensor<2x3xf32>) -> tensor<2x2xf32> {
    %weight = "mini.constant"() {value = dense<[[1.0, -2.0, 0.5], [0.25, 0.0, -0.25]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %bias = "mini.constant"() {value = dense<[0.5, -0.5]> : tensor<2xf32>} : () -> tensor<2xf32>
    %0 = "mini.linear"(%arg0, %weight, %bias) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

// CHECK-LABEL: func.func @quantize_linear_weight
// CHECK: %[[QW:.*]] = "mini.constant"() {value = dense<{{.*}}> : tensor<2x3xi8>} : () -> tensor<2x3xi8>
// CHECK: %[[BIAS:.*]] = "mini.constant"() {value = dense<[5.000000e-01, -5.000000e-01]> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: %[[OUT:.*]] = "mini.qlinear"(%arg0, %[[QW]], %[[BIAS]]) {weight_scale = {{.*}} : f32} : (tensor<2x3xf32>, tensor<2x3xi8>, tensor<2xf32>) -> tensor<2x2xf32>
// CHECK: return %[[OUT]]
