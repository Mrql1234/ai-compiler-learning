// RUN: %mini_compiler_opt --mini-const-fold %s | FileCheck %s

module {
  func.func @const_fold_relu() -> tensor<2xf32> {
    %0 = "mini.constant"() {value = dense<[-1.0, 2.5]> : tensor<2xf32>} : () -> tensor<2xf32>
    %1 = "mini.relu"(%0) : (tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }
}

// CHECK: func.func @const_fold_relu
// CHECK: "mini.constant"() {value = dense<[0.000000e+00, 2.500000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-NOT: "mini.relu"
