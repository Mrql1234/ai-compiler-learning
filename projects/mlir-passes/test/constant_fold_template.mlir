// 模板化常量折叠 Pass 测试
// 运行：./build-wsl/bin/mlir-passes-opt --constant-fold-template test/constant_fold_template.mlir
// RUN: mlir-passes-opt --constant-fold-template %s | FileCheck %s

func.func @test_template_integer_ops() -> i32 {
  %0 = arith.constant 50 : i32
  %1 = arith.constant 8 : i32
  %2 = arith.subi %0, %1 : i32
  return %2 : i32
}
// CHECK-LABEL: func.func @test_template_integer_ops
// CHECK: arith.constant 42
// CHECK-NOT: arith.subi

func.func @test_template_float_ops() -> f32 {
  %0 = arith.constant 6.0 : f32
  %1 = arith.constant 7.0 : f32
  %2 = arith.mulf %0, %1 : f32
  return %2 : f32
}
// CHECK-LABEL: func.func @test_template_float_ops
// CHECK: arith.constant 4.200000e+01 : f32
// CHECK-NOT: arith.mulf

func.func @test_template_chain() -> i32 {
  %0 = arith.constant 8 : i32
  %1 = arith.constant 30 : i32
  %2 = arith.constant 4 : i32
  %3 = arith.addi %0, %1 : i32
  %4 = arith.addi %3, %2 : i32
  return %4 : i32
}
// CHECK-LABEL: func.func @test_template_chain
// CHECK: arith.constant 42
// CHECK-NOT: arith.addi
