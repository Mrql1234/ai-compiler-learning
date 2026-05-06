// RUN: %mini_compiler_opt --mini-gpu-prep %s | FileCheck %s --check-prefix=PREP
// RUN: %mini_compiler_opt --mini-gpu-lowering %s | FileCheck %s --check-prefix=LOWER

module {
  func.func @prep_linear_relu(%arg0: tensor<2x4xf32>) -> tensor<2x8xf32> {
    %w = "mini.constant"() {value = dense<0.0> : tensor<8x4xf32>} : () -> tensor<8x4xf32>
    %b = "mini.constant"() {value = dense<0.0> : tensor<8xf32>} : () -> tensor<8xf32>
    %0 = "mini.linear"(%arg0, %w, %b) : (tensor<2x4xf32>, tensor<8x4xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %1 = "mini.relu"(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }
}

// PREP: func.func @prep_linear_relu
// PREP: linalg.fill
// PREP: linalg.matmul
// PREP: linalg.generic
// PREP-NOT: "mini.linear"
// PREP-NOT: "mini.relu"

// LOWER: module attributes {gpu.container_module}
// LOWER: gpu.launch_func
// LOWER: gpu.module
// LOWER: gpu.func
// LOWER-NOT: "mini.linear"
// LOWER-NOT: "mini.relu"
