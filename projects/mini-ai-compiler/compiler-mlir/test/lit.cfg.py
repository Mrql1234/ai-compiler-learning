# -*- Python -*-

import os

import lit.formats

from lit.llvm import llvm_config


config.name = "MINI_COMPILER_MLIR"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.mini_compiler_obj_root, "test")

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "CMakeLists.txt", "cpu_runner_demo.mlir", "lit.cfg.py", "lit.site.cfg.py.in"]
config.substitutions.append(
    ("%mini_compiler_opt", os.path.join(config.mini_compiler_obj_root, "bin", "mini-compiler-opt"))
)
config.substitutions.append(
    ("%mini_compiler_gpu_runner", os.path.join(config.mini_compiler_obj_root, "bin", "mini-compiler-gpu-runner"))
)

tool_dirs = [os.path.join(config.mini_compiler_obj_root, "bin"), config.llvm_tools_dir]
tools = [
    "FileCheck",
    "mini-compiler-opt",
    "mini-compiler-gpu-runner",
    "not",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
