// MLIR 内建 canonicalization / folding 测试
// 运行：./build-wsl/bin/mlir-passes-opt --constant-fold-builtin test/constant_fold_builtin.mlir
// RUN: mlir-passes-opt --constant-fold-builtin %s | FileCheck %s

func.func @test_builtin_integer_ops() -> i32 {
  %0 = arith.constant 21 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.muli %0, %1 : i32
  return %2 : i32
}
// CHECK-LABEL: func.func @test_builtin_integer_ops
// CHECK: arith.constant 42
// CHECK-NOT: arith.muli

func.func @test_builtin_float_ops() -> f32 {
  %0 = arith.constant 40.0 : f32
  %1 = arith.constant 2.0 : f32
  %2 = arith.addf %0, %1 : f32
  return %2 : f32
}
// CHECK-LABEL: func.func @test_builtin_float_ops
// CHECK: arith.constant 4.200000e+01 : f32
// CHECK-NOT: arith.addf

func.func @test_builtin_chain() -> i32 {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 39 : i32
  %3 = arith.addi %0, %1 : i32
  %4 = arith.addi %3, %2 : i32
  return %4 : i32
}
// CHECK-LABEL: func.func @test_builtin_chain
// CHECK: arith.constant 42
// CHECK-NOT: arith.addi
