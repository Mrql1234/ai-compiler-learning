configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
)

set(MINI_COMPILER_MLIR_TEST_DEPENDS
  mini-compiler-opt
)

add_lit_testsuite(check-mini-compiler-mlir "Running mini compiler MLIR tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MINI_COMPILER_MLIR_TEST_DEPENDS}
)
