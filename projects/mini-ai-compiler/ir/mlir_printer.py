from __future__ import annotations

from typing import Any

import numpy as np

from ir.graph import Graph
from ir.node import Node
from ir.value import Value


def print_mlir(graph: Graph, *, generic_ops: bool = False) -> str:
    lines = []
    signature = ", ".join(_format_block_arg(index, value) for index, value in enumerate(graph.inputs))
    lines.append(f"module {{")
    lines.append(f"  func.func @{graph.name}({signature}) -> {_format_return_types(graph.outputs)} {{")

    input_names = {id(value): f"%arg{index}" for index, value in enumerate(graph.inputs)}
    names = dict(input_names)

    for node in graph.nodes:
        result_names = [_name_for_value(value, names) for value in node.outputs]
        operand_names = [_name_for_value(value, names) for value in node.inputs]
        lines.extend(_print_node(node, result_names, operand_names, generic_ops=generic_ops))

    return_operands = ", ".join(_name_for_value(value, names) for value in graph.outputs)
    return_types = ", ".join(_format_value_type(value) for value in graph.outputs)
    lines.append(f"    return {return_operands} : {return_types}")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def _print_node(
    node: Node,
    result_names: list[str],
    operand_names: list[str],
    *,
    generic_ops: bool = False,
) -> list[str]:
    result_prefix = ", ".join(result_names)
    result_suffix = _format_result_types(node.outputs)
    op_name = f"mini.{node.op_type}"
    generic_op_name = f'"{op_name}"'

    if node.op_type == "constant":
        value = node.attrs["value"]
        attr_text = _format_dense_attr(value)
        if generic_ops:
            return [
                f"    {result_prefix} = {generic_op_name}() "
                f"{{value = {attr_text} : {result_suffix}}} : () -> {result_suffix}"
            ]
        return [f"    {result_prefix} = {op_name} {attr_text} : {result_suffix}"]

    if node.attrs:
        attrs = ", ".join(f"{key} = {_format_attr_value(value)}" for key, value in node.attrs.items())
        attr_text = f" {{{attrs}}}"
    else:
        attr_text = ""

    operand_text = ", ".join(operand_names)
    operand_types = ", ".join(_format_value_type(value) for value in node.inputs)
    if generic_ops:
        return [
            f"    {result_prefix} = {generic_op_name}({operand_text}){attr_text} : ({operand_types}) -> {result_suffix}"
        ]
    return [
        f"    {result_prefix} = {op_name} {operand_text}{attr_text} : ({operand_types}) -> {result_suffix}"
    ]


def _name_for_value(value: Value, names: dict[int, str]) -> str:
    key = id(value)
    if key not in names:
        names[key] = f"%{value.name}"
    return names[key]


def _format_block_arg(index: int, value: Value) -> str:
    return f"%arg{index}: {_format_value_type(value)}"


def _format_return_types(values: list[Value]) -> str:
    if len(values) == 1:
        return _format_value_type(values[0])
    return f"({', '.join(_format_value_type(value) for value in values)})"


def _format_result_types(values: list[Value]) -> str:
    if len(values) == 1:
        return _format_value_type(values[0])
    return f"({', '.join(_format_value_type(value) for value in values)})"


def _format_value_type(value: Value) -> str:
    if value.type is None or value.type.shape is None:
        return "tensor<*xf32>"
    dtype = _map_dtype(value.type.dtype)
    shape = "x".join("?" if dim == -1 else str(dim) for dim in value.type.shape)
    if shape == "":
        return f"tensor<{dtype}>"
    return f"tensor<{shape}x{dtype}>"


def _map_dtype(dtype: str | None) -> str:
    if dtype is None:
        return "f32"
    text = dtype.lower()
    if "float64" in text or text == "double":
        return "f64"
    if "float16" in text or text == "half":
        return "f16"
    if "float32" in text or text in {"float", "torch.float32"}:
        return "f32"
    if "int64" in text:
        return "i64"
    if "int32" in text:
        return "i32"
    if "bool" in text:
        return "i1"
    return text


def _format_dense_attr(value: Any) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        return f"dense<{scalar}>"
    return f"dense<{array.tolist()}>"


def _format_attr_value(value: Any) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return "[" + ", ".join(_format_attr_value(item) for item in value) + "]"
    return str(value)
