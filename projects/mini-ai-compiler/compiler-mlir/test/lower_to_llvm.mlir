// RUN: %mini_compiler_opt \
// RUN:   --mini-lower-to-linalg \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
// RUN:   --drop-equivalent-buffer-results \
// RUN:   --buffer-results-to-out-params \
// RUN:   --convert-bufferization-to-memref \
// RUN:   --convert-linalg-to-loops \
// RUN:   --convert-scf-to-cf \
// RUN:   --convert-cf-to-llvm \
// RUN:   --convert-arith-to-llvm \
// RUN:   --convert-index-to-llvm \
// RUN:   --expand-realloc \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts %s | FileCheck %s

module {
  func.func @lower_to_llvm(%arg0: tensor<2x4xf32>) -> tensor<2x8xf32> {
    %w = "mini.constant"() {value = dense<0.0> : tensor<8x4xf32>} : () -> tensor<8x4xf32>
    %b = "mini.constant"() {value = dense<0.0> : tensor<8xf32>} : () -> tensor<8xf32>
    %0 = "mini.linear"(%arg0, %w, %b) : (tensor<2x4xf32>, tensor<8x4xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %1 = "mini.relu"(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }
}

// CHECK: llvm.func @malloc
// CHECK: llvm.mlir.global private constant
// CHECK: llvm.func @lower_to_llvm
// CHECK: llvm.call @malloc
// CHECK: llvm.fmul
// CHECK: llvm.fadd
// CHECK-NOT: "mini.linear"
// CHECK-NOT: "mini.relu"
