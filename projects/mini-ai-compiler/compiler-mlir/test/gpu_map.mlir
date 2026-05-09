// RUN: %mini_compiler_opt %s --mini-gpu-map | FileCheck %s

func.func @map_demo(%lb: index, %ub: index, %step: index) {
  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%step, %step) {
    scf.parallel (%k) = (%lb) to (%ub) step (%step) {
      scf.reduce
    }
    scf.reduce
  }
  return
}

// CHECK: func.func @map_demo
// CHECK-DAG: mapping = [#gpu.loop_dim_map<processor = block_y
// CHECK-DAG: #gpu.loop_dim_map<processor = block_x
// CHECK-DAG: mapping = [#gpu.loop_dim_map<processor = thread_x
