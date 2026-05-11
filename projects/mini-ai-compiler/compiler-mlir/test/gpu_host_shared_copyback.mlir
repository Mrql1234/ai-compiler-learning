// RUN: %mini_compiler_opt %s --mini-gpu-host-shared | FileCheck %s

module attributes {gpu.container_module} {
  func.func @copyback_demo(%arg0: memref<2x2xf32>) {
    %c1 = arith.constant 1 : index
    gpu.launch_func @copyback_kernel::@copyback_kernel
        blocks in (%c1, %c1, %c1)
        threads in (%c1, %c1, %c1)
        args(%arg0 : memref<2x2xf32>)
    return
  }

  gpu.module @copyback_kernel {
    gpu.func @copyback_kernel(%arg0: memref<2x2xf32>) kernel {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.000000e+00 : f32
      memref.store %cst, %arg0[%c0, %c0] : memref<2x2xf32>
      gpu.return
    }
  }
}

// CHECK: %[[SHARED:.*]] = gpu.alloc  host_shared () : memref<2x2xf32>
// CHECK: memref.copy %arg0, %[[SHARED]] : memref<2x2xf32> to memref<2x2xf32>
// CHECK: gpu.launch_func
// CHECK-SAME: args(%[[SHARED]] : memref<2x2xf32>)
// CHECK: memref.copy %[[SHARED]], %arg0 : memref<2x2xf32> to memref<2x2xf32>
