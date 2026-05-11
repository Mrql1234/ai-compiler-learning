// RUN: %mini_compiler_opt --mini-quantized-gpu-lowering %s | FileCheck %s

module {
  func.func @gpu_quant_kernel(%arg0: tensor<2x3xf32>) -> tensor<2x2xf32> {
    %weight = "mini.constant"() {value = dense<[[1.0, -2.0, 0.5], [0.25, 0.0, -0.25]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %bias = "mini.constant"() {value = dense<[0.5, -0.5]> : tensor<2xf32>} : () -> tensor<2xf32>
    %0 = "mini.linear"(%arg0, %weight, %bias) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

// CHECK: module attributes {gpu.container_module}
// CHECK: gpu.launch_func
// CHECK: gpu.module
// CHECK: gpu.func
// CHECK: scf.for
// CHECK: memref.load {{.*}} : memref<2x3xi8>
// CHECK: arith.sitofp
// CHECK: arith.mulf
// CHECK: arith.addf
