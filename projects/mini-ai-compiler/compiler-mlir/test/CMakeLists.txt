configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MINI_COMPILER_MLIR_TEST_DEPENDS
  FileCheck
  mini-compiler-opt
  mini-compiler-gpu-runner
)

add_lit_testsuite(check-mini-compiler-mlir "Running mini compiler MLIR tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MINI_COMPILER_MLIR_TEST_DEPENDS}
)

add_lit_testsuites(MINI_COMPILER_MLIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MINI_COMPILER_MLIR_TEST_DEPENDS}
)

add_test(
  NAME mini-compiler-mlir-lit
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-lit -sv ${CMAKE_CURRENT_BINARY_DIR}
)
