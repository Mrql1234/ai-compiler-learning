// RUN: %mini_compiler_opt %s --mini-gpu-tile --mini-gpu-map | FileCheck %s

func.func @tile_map_demo(%lb: index, %ub0: index, %ub1: index, %step: index) {
  scf.parallel (%i, %j) = (%lb, %lb) to (%ub0, %ub1) step (%step, %step) {
    scf.reduce
  }
  return
}

// CHECK: func.func @tile_map_demo
// CHECK: scf.parallel
// CHECK-DAG: mapping = [#gpu.loop_dim_map<processor = block_y
// CHECK-DAG: #gpu.loop_dim_map<processor = block_x
// CHECK-DAG: mapping = [#gpu.loop_dim_map<processor = thread_y
// CHECK-DAG: #gpu.loop_dim_map<processor = thread_x
// CHECK-DAG: arith.cmpi ult
// CHECK-DAG: scf.if
