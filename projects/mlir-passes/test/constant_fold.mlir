// 常量折叠 Pass 测试
// 运行：mlir-opt --constant-fold constant_fold.mlir

// RUN: mlir-opt --constant-fold %s | FileCheck %s

func.func @test_add_constant() -> i32 {
  // 优化前：
  // %0 = arith.constant 42 : i32
  // %1 = arith.constant 0 : i32
  // %2 = arith.addi %0, %1 : i32
  // return %2 : i32
  
  %0 = arith.constant 42 : i32
  %1 = arith.constant 0 : i32
  %2 = arith.addi %0, %1 : i32
  return %2 : i32
}
// CHECK-LABEL: func.func @test_add_constant
// CHECK: arith.constant 42
// CHECK-NOT: arith.addi

func.func @test_mul_constant() -> i32 {
  %0 = arith.constant 6 : i32
  %1 = arith.constant 7 : i32
  %2 = arith.muli %0, %1 : i32
  return %2 : i32
}
// CHECK-LABEL: func.func @test_mul_constant
// CHECK: arith.constant 42
// CHECK-NOT: arith.muli

func.func @test_float_add() -> f32 {
  %0 = arith.constant 1.5 : f32
  %1 = arith.constant 2.5 : f32
  %2 = arith.addf %0, %1 : f32
  return %2 : f32
}
// CHECK-LABEL: func.func @test_float_add
// CHECK: arith.constant 4.0
// CHECK-NOT: arith.addf

// 不能折叠的情况：非常量操作数
func.func @test_non_constant(%arg0: i32) -> i32 {
  %0 = arith.constant 42 : i32
  %1 = arith.addi %arg0, %0 : i32
  return %1 : i32
}
// CHECK-LABEL: func.func @test_non_constant
// CHECK: arith.addi
