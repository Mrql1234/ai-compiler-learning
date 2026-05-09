// RUN: %mini_compiler_opt --mini-fusion %s | FileCheck %s --check-prefix=FUSE
// RUN: %mini_compiler_opt --mini-gpu-prep %s | FileCheck %s --check-prefix=PREP
// RUN: %mini_compiler_opt --mini-gpu-lowering %s | FileCheck %s --check-prefix=LOWER

module {
  func.func @fuse_matmul_add_relu(%arg0: tensor<2x4xf32>) -> tensor<2x8xf32> {
    %w = "mini.constant"() {value = dense<0.0> : tensor<4x8xf32>} : () -> tensor<4x8xf32>
    %b = "mini.constant"() {value = dense<0.0> : tensor<8xf32>} : () -> tensor<8xf32>
    %0 = "mini.matmul"(%arg0, %w) : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = "mini.add"(%0, %b) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %2 = "mini.relu"(%1) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    return %2 : tensor<2x8xf32>
  }
}

// FUSE: func.func @fuse_matmul_add_relu
// FUSE: "mini.fused_matmul_add_relu"
// FUSE-NOT: "mini.matmul"
// FUSE-NOT: "mini.add"
// FUSE-NOT: "mini.relu"

// PREP: func.func @fuse_matmul_add_relu
// PREP: linalg.fill
// PREP: linalg.matmul
// PREP-COUNT-2: linalg.generic
// PREP-NOT: "mini.matmul"
// PREP-NOT: "mini.add"
// PREP-NOT: "mini.relu"

// LOWER: module attributes {gpu.container_module}
// LOWER: gpu.launch_func
// LOWER: gpu.module
// LOWER: gpu.func
// LOWER-NOT: "mini.matmul"
// LOWER-NOT: "mini.add"
// LOWER-NOT: "mini.relu"
