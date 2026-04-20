// 死代码消除 Pass 测试
// 运行：mlir-opt --dead-code-elim dce.mlir

// RUN: mlir-opt --dead-code-elim %s | FileCheck %s

func.func @test_unused_result() -> i32 {
  // 优化前：
  // %0 = arith.constant 42 : i32
  // %1 = arith.mulf %0, %0 : f32  // 未使用
  // %2 = arith.addi %0, %0 : i32
  // return %2 : i32
  
  %0 = arith.constant 42 : i32
  %1 = arith.constant 2.0 : f32
  %2 = "math.exp"(%1) : (f32) -> f32  // 未使用
  %3 = arith.addi %0, %0 : i32
  return %3 : i32
}
// CHECK-LABEL: func.func @test_unused_result
// CHECK: arith.constant 42
// CHECK: arith.addi
// CHECK-NOT: math.exp
// CHECK: return

func.func @test_chain_elim() -> i32 {
  // 链式死代码
  %0 = arith.constant 1 : i32
  %1 = arith.addi %0, %0 : i32  // 未使用
  %2 = arith.muli %1, %1 : i32  // 未使用
  %3 = arith.constant 42 : i32
  return %3 : i32
}
// CHECK-LABEL: func.func @test_chain_elim
// CHECK: arith.constant 42
// CHECK-NOT: arith.addi
// CHECK-NOT: arith.muli
// CHECK: return
