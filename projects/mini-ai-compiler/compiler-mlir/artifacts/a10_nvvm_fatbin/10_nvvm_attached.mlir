#map = affine_map<()[s0, s1, s2] -> (s0 * s2 + s1)>
module attributes {gpu.container_module} {
  memref.global "private" constant @__constant_8xf32 : memref<8xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x4xf32 : memref<8x4xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  func.func @prep_linear_relu(%arg0: memref<2x4xf32>) -> memref<2x8xf32> {
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x8xf32>
    gpu.launch_func  @prep_linear_relu_kernel::@prep_linear_relu_kernel blocks in (%c2, %c8, %c1) threads in (%c1, %c1, %c1)  args(%c1 : index, %c0 : index, %cst : f32, %alloc : memref<2x8xf32>)
    gpu.launch_func  @prep_linear_relu_kernel_0::@prep_linear_relu_kernel blocks in (%c2, %c8, %c1) threads in (%c1, %c1, %c1)  args(%c1 : index, %c0 : index, %arg0 : memref<2x4xf32>, %alloc : memref<2x8xf32>, %cst : f32, %c4 : index)
    gpu.launch_func  @prep_linear_relu_kernel_1::@prep_linear_relu_kernel blocks in (%c2, %c8, %c1) threads in (%c1, %c1, %c1)  args(%c1 : index, %c0 : index, %alloc : memref<2x8xf32>, %cst : f32)
    gpu.launch_func  @prep_linear_relu_kernel_2::@prep_linear_relu_kernel blocks in (%c2, %c8, %c1) threads in (%c1, %c1, %c1)  args(%c1 : index, %c0 : index, %alloc : memref<2x8xf32>, %cst : f32)
    return %alloc : memref<2x8xf32>
  }
  gpu.module @prep_linear_relu_kernel [#nvvm.target<O = 3, chip = "sm_86">] {
    gpu.func @prep_linear_relu_kernel(%arg0: index, %arg1: index, %arg2: f32, %arg3: memref<2x8xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = affine.apply #map()[%arg0, %arg1, %block_id_x]
      %1 = affine.apply #map()[%arg0, %arg1, %block_id_y]
      memref.store %arg2, %arg3[%0, %1] : memref<2x8xf32>
      gpu.return
    }
  }
  gpu.module @prep_linear_relu_kernel_0 [#nvvm.target<O = 3, chip = "sm_86">] {
    gpu.func @prep_linear_relu_kernel(%arg0: index, %arg1: index, %arg2: memref<2x4xf32>, %arg3: memref<2x8xf32>, %arg4: f32, %arg5: index) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = affine.apply #map()[%arg0, %arg1, %block_id_x]
      %1 = affine.apply #map()[%arg0, %arg1, %block_id_y]
      scf.for %arg6 = %arg1 to %arg5 step %arg0 {
        %2 = memref.load %arg2[%0, %arg6] : memref<2x4xf32>
        %3 = memref.load %arg3[%0, %1] : memref<2x8xf32>
        %4 = arith.mulf %2, %arg4 : f32
        %5 = arith.addf %3, %4 : f32
        memref.store %5, %arg3[%0, %1] : memref<2x8xf32>
      }
      gpu.return
    }
  }
  gpu.module @prep_linear_relu_kernel_1 [#nvvm.target<O = 3, chip = "sm_86">] {
    gpu.func @prep_linear_relu_kernel(%arg0: index, %arg1: index, %arg2: memref<2x8xf32>, %arg3: f32) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = affine.apply #map()[%arg0, %arg1, %block_id_x]
      %1 = affine.apply #map()[%arg0, %arg1, %block_id_y]
      %2 = memref.load %arg2[%0, %1] : memref<2x8xf32>
      %3 = arith.addf %2, %arg3 : f32
      memref.store %3, %arg2[%0, %1] : memref<2x8xf32>
      gpu.return
    }
  }
  gpu.module @prep_linear_relu_kernel_2 [#nvvm.target<O = 3, chip = "sm_86">] {
    gpu.func @prep_linear_relu_kernel(%arg0: index, %arg1: index, %arg2: memref<2x8xf32>, %arg3: f32) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = affine.apply #map()[%arg0, %arg1, %block_id_x]
      %1 = affine.apply #map()[%arg0, %arg1, %block_id_y]
      %2 = memref.load %arg2[%0, %1] : memref<2x8xf32>
      %3 = arith.maximumf %2, %arg3 : f32
      memref.store %3, %arg2[%0, %1] : memref<2x8xf32>
      gpu.return
    }
  }
}

