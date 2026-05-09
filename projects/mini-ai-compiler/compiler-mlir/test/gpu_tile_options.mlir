// RUN: %mini_compiler_opt %s --mini-gpu-tile-pipeline="tile-sizes=4,2" | FileCheck %s

func.func @tile_options_demo(%lb: index, %ub0: index, %ub1: index, %step: index) {
  scf.parallel (%i, %j) = (%lb, %lb) to (%ub0, %ub1) step (%step, %step) {
    scf.reduce
  }
  return
}

// CHECK: %c4 = arith.constant 4 : index
// CHECK: %c2 = arith.constant 2 : index
// CHECK: %[[STEP0:.*]] = arith.muli %arg3, %c4 : index
// CHECK: %[[STEP1:.*]] = arith.muli %arg3, %c2 : index
// CHECK: scf.parallel
// CHECK-SAME: step (%[[STEP0]], %[[STEP1]])
