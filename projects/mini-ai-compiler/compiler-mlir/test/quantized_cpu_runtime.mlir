// RUN: %mini_compiler_opt --pass-pipeline='builtin.module(func.func(mini-canonicalize,mini-fusion,mini-quantize-weights),mini-lower-quant-to-cpu-runtime)' %s | FileCheck %s

module {
  func.func @cpu_quant_runtime(%arg0: tensor<2x3xf32>) -> tensor<2x2xf32> {
    %weight = "mini.constant"() {value = dense<[[1.0, -2.0, 0.5], [0.25, 0.0, -0.25]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %bias = "mini.constant"() {value = dense<[0.5, -0.5]> : tensor<2xf32>} : () -> tensor<2xf32>
    %0 = "mini.linear"(%arg0, %weight, %bias) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

// CHECK-LABEL: func.func @cpu_quant_runtime
// CHECK: %[[SCALE:.*]] = arith.constant {{.*}} : f32
// CHECK: %[[QW:.*]] = arith.constant dense<{{.*}}> : tensor<2x3xi8>
// CHECK: %[[OUT:.*]] = call @__mini_cpu_qlinear_runtime_
// CHECK-LABEL: func.func private @__mini_cpu_qlinear_runtime_
// CHECK: arith.sitofp
// CHECK: arith.mulf
// CHECK: arith.addf
