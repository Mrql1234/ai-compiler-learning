// RUN: not %mini_compiler_gpu_runner %s --kernel-backend=cuda_hand 2>&1 | FileCheck %s --check-prefix=CUDA-HAND
// RUN: not %mini_compiler_gpu_runner %s --kernel-backend=cublas 2>&1 | FileCheck %s --check-prefix=CUBLAS
// CUDA-HAND: kernel backend 'cuda_hand' is recognized but not implemented
// CUBLAS: kernel backend 'cublas' is recognized but not implemented

module {
  func.func @run() -> f32 attributes {llvm.emit_c_interface} {
    %c = arith.constant 0.0 : f32
    return %c : f32
  }
}
