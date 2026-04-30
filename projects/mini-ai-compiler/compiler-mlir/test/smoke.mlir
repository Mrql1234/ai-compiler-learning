// RUN: %mini_compiler_opt %s | FileCheck %s

module {
  func.func @smoke(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "mini.relu"(%arg0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}

// CHECK: func.func @smoke
