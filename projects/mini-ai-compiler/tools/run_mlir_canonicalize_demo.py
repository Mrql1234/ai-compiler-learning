from __future__ import annotations

import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from ir.graph import Graph
from ir.node import Node
from ir.printer import print_graph
from passes.mlir_canonicalize import MLIRCanonicalizePass


def build_demo_graph() -> Graph:
    graph = Graph(name="canonicalize_demo")
    x = graph.add_input("x")
    c0 = graph.add_constant("c0", np.array(0, dtype=np.int32))
    c1 = graph.add_constant("c1", np.array(1, dtype=np.int32))
    c2 = graph.add_constant("c2", np.array(2, dtype=np.int32))
    c3 = graph.add_constant("c3", np.array(3, dtype=np.int32))

    add_out = graph.new_value("add_out")
    mul_out = graph.new_value("mul_out")
    passthrough_out = graph.new_value("passthrough_out")
    final_out = graph.new_value("final_out")

    graph.add_node(Node("add_consts", "add", [c2, c3], [add_out]))
    graph.add_node(Node("mul_consts", "mul", [add_out, c1], [mul_out]))
    graph.add_node(Node("add_zero", "add", [x, c0], [passthrough_out]))
    graph.add_node(Node("final_add", "add", [passthrough_out, mul_out], [final_out]))
    graph.outputs = [final_out]
    return graph


def main() -> None:
    graph = build_demo_graph()
    print("=== Before MLIR-style Canonicalize ===")
    print(print_graph(graph))

    MLIRCanonicalizePass().run(graph)

    print("\n=== After MLIR-style Canonicalize ===")
    print(print_graph(graph))


if __name__ == "__main__":
    main()
