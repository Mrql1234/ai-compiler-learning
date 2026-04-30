// 增强版手写常量折叠 Pass 测试
// 运行：./build-wsl/bin/mlir-passes-opt --constant-fold-extended test/constant_fold_extended.mlir
// RUN: mlir-passes-opt --constant-fold-extended %s | FileCheck %s

func.func @test_sub_constant() -> i32 {
  %0 = arith.constant 50 : i32
  %1 = arith.constant 8 : i32
  %2 = arith.subi %0, %1 : i32
  return %2 : i32
}
// CHECK-LABEL: func.func @test_sub_constant
// CHECK: arith.constant 42
// CHECK-NOT: arith.subi

func.func @test_float_sub() -> f32 {
  %0 = arith.constant 43.5 : f32
  %1 = arith.constant 1.5 : f32
  %2 = arith.subf %0, %1 : f32
  return %2 : f32
}
// CHECK-LABEL: func.func @test_float_sub
// CHECK: arith.constant 4.200000e+01 : f32
// CHECK-NOT: arith.subf

func.func @test_float_mul() -> f32 {
  %0 = arith.constant 6.0 : f32
  %1 = arith.constant 7.0 : f32
  %2 = arith.mulf %0, %1 : f32
  return %2 : f32
}
// CHECK-LABEL: func.func @test_float_mul
// CHECK: arith.constant 4.200000e+01 : f32
// CHECK-NOT: arith.mulf
