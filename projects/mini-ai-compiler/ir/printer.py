from __future__ import annotations

from pathlib import Path

from ir.graph import Graph
from ir.mlir_printer import print_mlir


def print_graph(graph: Graph) -> str:
    return str(graph)


def write_graph(graph: Graph, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(str(graph), encoding="utf-8")
    return output_path


def write_mlir(graph: Graph, path: str | Path, *, generic_ops: bool = False) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(print_mlir(graph, generic_ops=generic_ops), encoding="utf-8")
    return output_path
