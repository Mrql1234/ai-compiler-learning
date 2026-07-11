#map = affine_map<()[s0, s1, s2] -> (s0 * s2 + s1)>
module attributes {gpu.container_module} {
  memref.global "private" constant @__constant_2x4xf32 : memref<2x4xf32> = dense<[[1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8xf32 : memref<8xf32> = dense<[5.000000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x4xf32 : memref<8x4xf32> = dense<[[1.000000e+00, 2.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> {alignment = 64 : i64}
  func.func @compute(%arg0: memref<2x4xf32>) -> memref<2x8xf32> {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_8x4xf32 : memref<8x4xf32>
    %1 = memref.get_global @__constant_8xf32 : memref<8xf32>
    %memref = gpu.alloc  host_shared () : memref<2x8xf32>
    gpu.launch_func  @compute_kernel::@compute_kernel blocks in (%c1, %c1, %c1) threads in (%c8, %c8, %c1)  args(%c8 : index, %c0 : index, %c8 : index, %c1 : index, %c0 : index, %c2 : index, %c8 : index, %cst : f32, %memref : memref<2x8xf32>)
    %memref_0 = gpu.alloc  host_shared () : memref<2x4xf32>
    memref.copy %arg0, %memref_0 : memref<2x4xf32> to memref<2x4xf32>
    %memref_1 = gpu.alloc  host_shared () : memref<8x4xf32>
    memref.copy %0, %memref_1 : memref<8x4xf32> to memref<8x4xf32>
    gpu.launch_func  @compute_kernel_0::@compute_kernel blocks in (%c1, %c1, %c1) threads in (%c8, %c8, %c1)  args(%c8 : index, %c0 : index, %c8 : index, %c1 : index, %c0 : index, %c2 : index, %c8 : index, %memref_0 : memref<2x4xf32>, %memref_1 : memref<8x4xf32>, %memref : memref<2x8xf32>, %c4 : index)
    %memref_2 = gpu.alloc  host_shared () : memref<8xf32>
    memref.copy %1, %memref_2 : memref<8xf32> to memref<8xf32>
    gpu.launch_func  @compute_kernel_1::@compute_kernel blocks in (%c1, %c1, %c1) threads in (%c8, %c8, %c1)  args(%c8 : index, %c0 : index, %c8 : index, %c1 : index, %c0 : index, %c2 : index, %c8 : index, %memref : memref<2x8xf32>, %memref_2 : memref<8xf32>)
    gpu.launch_func  @compute_kernel_2::@compute_kernel blocks in (%c1, %c1, %c1) threads in (%c8, %c8, %c1)  args(%c8 : index, %c0 : index, %c8 : index, %c1 : index, %c0 : index, %c2 : index, %c8 : index, %memref : memref<2x8xf32>, %cst : f32)
    return %memref : memref<2x8xf32>
  }
  gpu.module @compute_kernel {
    gpu.func @compute_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: f32, %arg8: memref<2x8xf32>) kernel {
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %thread_id_x = gpu.thread_id x
      %thread_id_y = gpu.thread_id y
      %0 = affine.apply #map()[%arg0, %arg1, %block_id_y]
      %1 = affine.apply #map()[%arg2, %arg1, %block_id_x]
      %2 = affine.apply #map()[%arg3, %arg4, %thread_id_y]
      %3 = affine.apply #map()[%arg3, %arg4, %thread_id_x]
      %4 = arith.addi %2, %0 : index
      %5 = arith.addi %3, %1 : index
      %6 = arith.muli %2, %arg3 : index
      %7 = arith.addi %6, %0 : index
      %8 = arith.cmpi ult, %7, %arg5 : index
      %9 = arith.muli %3, %arg3 : index
      %10 = arith.addi %9, %1 : index
      %11 = arith.cmpi ult, %10, %arg6 : index
      %12 = arith.andi %8, %11 : i1
      scf.if %12 {
        memref.store %arg7, %arg8[%4, %5] : memref<2x8xf32>
      }
      gpu.return
    }
  }
  gpu.module @compute_kernel_0 {
    gpu.func @compute_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: memref<2x4xf32>, %arg8: memref<8x4xf32>, %arg9: memref<2x8xf32>, %arg10: index) kernel {
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %thread_id_x = gpu.thread_id x
      %thread_id_y = gpu.thread_id y
      %0 = affine.apply #map()[%arg0, %arg1, %block_id_y]
      %1 = affine.apply #map()[%arg2, %arg1, %block_id_x]
      %2 = affine.apply #map()[%arg3, %arg4, %thread_id_y]
      %3 = affine.apply #map()[%arg3, %arg4, %thread_id_x]
      %4 = arith.addi %2, %0 : index
      %5 = arith.addi %3, %1 : index
      %6 = arith.muli %2, %arg3 : index
      %7 = arith.addi %6, %0 : index
      %8 = arith.cmpi ult, %7, %arg5 : index
      %9 = arith.muli %3, %arg3 : index
      %10 = arith.addi %9, %1 : index
      %11 = arith.cmpi ult, %10, %arg6 : index
      %12 = arith.andi %8, %11 : i1
      scf.if %12 {
        scf.for %arg11 = %arg1 to %arg10 step %arg3 {
          %13 = memref.load %arg7[%4, %arg11] : memref<2x4xf32>
          %14 = memref.load %arg8[%5, %arg11] : memref<8x4xf32>
          %15 = memref.load %arg9[%4, %5] : memref<2x8xf32>
          %16 = arith.mulf %13, %14 : f32
          %17 = arith.addf %15, %16 : f32
          memref.store %17, %arg9[%4, %5] : memref<2x8xf32>
        }
      }
      gpu.return
    }
  }
  gpu.module @compute_kernel_1 {
    gpu.func @compute_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: memref<2x8xf32>, %arg8: memref<8xf32>) kernel {
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %thread_id_x = gpu.thread_id x
      %thread_id_y = gpu.thread_id y
      %0 = affine.apply #map()[%arg0, %arg1, %block_id_y]
      %1 = affine.apply #map()[%arg2, %arg1, %block_id_x]
      %2 = affine.apply #map()[%arg3, %arg4, %thread_id_y]
      %3 = affine.apply #map()[%arg3, %arg4, %thread_id_x]
      %4 = arith.addi %2, %0 : index
      %5 = arith.addi %3, %1 : index
      %6 = arith.muli %2, %arg3 : index
      %7 = arith.addi %6, %0 : index
      %8 = arith.cmpi ult, %7, %arg5 : index
      %9 = arith.muli %3, %arg3 : index
      %10 = arith.addi %9, %1 : index
      %11 = arith.cmpi ult, %10, %arg6 : index
      %12 = arith.andi %8, %11 : i1
      scf.if %12 {
        %13 = memref.load %arg7[%4, %5] : memref<2x8xf32>
        %14 = memref.load %arg8[%5] : memref<8xf32>
        %15 = arith.addf %13, %14 : f32
        memref.store %15, %arg7[%4, %5] : memref<2x8xf32>
      }
      gpu.return
    }
  }
  gpu.module @compute_kernel_2 {
    gpu.func @compute_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: memref<2x8xf32>, %arg8: f32) kernel {
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %thread_id_x = gpu.thread_id x
      %thread_id_y = gpu.thread_id y
      %0 = affine.apply #map()[%arg0, %arg1, %block_id_y]
      %1 = affine.apply #map()[%arg2, %arg1, %block_id_x]
      %2 = affine.apply #map()[%arg3, %arg4, %thread_id_y]
      %3 = affine.apply #map()[%arg3, %arg4, %thread_id_x]
      %4 = arith.addi %2, %0 : index
      %5 = arith.addi %3, %1 : index
      %6 = arith.muli %2, %arg3 : index
      %7 = arith.addi %6, %0 : index
      %8 = arith.cmpi ult, %7, %arg5 : index
      %9 = arith.muli %3, %arg3 : index
      %10 = arith.addi %9, %1 : index
      %11 = arith.cmpi ult, %10, %arg6 : index
      %12 = arith.andi %8, %11 : i1
      scf.if %12 {
        %13 = memref.load %arg7[%4, %5] : memref<2x8xf32>
        %14 = arith.maximumf %13, %arg8 : f32
        memref.store %14, %arg7[%4, %5] : memref<2x8xf32>
      }
      gpu.return
    }
  }
  func.func @run() -> f32 attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_2x4xf32 : memref<2x4xf32>
    %1 = call @compute(%0) : (memref<2x4xf32>) -> memref<2x8xf32>
    %2 = memref.load %1[%c0, %c0] : memref<2x8xf32>
    return %2 : f32
  }
}

