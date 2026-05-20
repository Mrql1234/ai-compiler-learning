// RUN: not %mini_compiler_gpu_runner %s --kernel-backend=cutlass 2>&1 | FileCheck %s --check-prefix=CUTLASS
// CUTLASS: kernel backend 'cutlass' is recognized but not implemented

module {
  func.func @run() -> f32 attributes {llvm.emit_c_interface} {
    %c = arith.constant 0.0 : f32
    return %c : f32
  }
}
