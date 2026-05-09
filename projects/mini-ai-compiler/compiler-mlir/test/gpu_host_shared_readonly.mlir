// RUN: %mini_compiler_opt %s --mini-gpu-host-shared | FileCheck %s

module attributes {gpu.container_module} {
  memref.global "private" constant @weights : memref<2x2xf32> = dense<1.0>

  func.func @readonly_demo() {
    %weights = memref.get_global @weights : memref<2x2xf32>
    %c1 = arith.constant 1 : index
    gpu.launch_func @readonly_kernel::@readonly_kernel
        blocks in (%c1, %c1, %c1)
        threads in (%c1, %c1, %c1)
        args(%weights : memref<2x2xf32>)
    return
  }

  gpu.module @readonly_kernel {
    gpu.func @readonly_kernel(%arg0: memref<2x2xf32>) kernel {
      gpu.return
    }
  }
}

// CHECK: %[[WEIGHTS:.*]] = memref.get_global @weights : memref<2x2xf32>
// CHECK: %[[SHARED:.*]] = gpu.alloc  host_shared () : memref<2x2xf32>
// CHECK: memref.copy %[[WEIGHTS]], %[[SHARED]] : memref<2x2xf32> to memref<2x2xf32>
// CHECK: gpu.launch_func
// CHECK-SAME: args(%[[SHARED]] : memref<2x2xf32>)
// CHECK-NOT: memref.copy %[[SHARED]], %[[WEIGHTS]]
