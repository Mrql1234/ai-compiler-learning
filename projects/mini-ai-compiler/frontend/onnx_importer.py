from __future__ import annotations

from typing import Any

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:  # pragma: no cover
    onnx = None
    numpy_helper = None

from ir.graph import Graph
from ir.node import Node
from ir.types import TensorType


class ONNXImporter:
    _op_map = {
        "MatMul": "matmul",
        "Add": "add",
        "Mul": "mul",
        "Relu": "relu",
    }

    def import_model(self, model_or_path: Any) -> Graph:
        if onnx is None or numpy_helper is None:
            raise RuntimeError("onnx is required for ONNX import.")

        model = onnx.load(model_or_path) if isinstance(model_or_path, str) else model_or_path
        graph = Graph(name=model.graph.name or "onnx_graph")
        env = {}

        for initializer in model.graph.initializer:
            array = numpy_helper.to_array(initializer)
            env[initializer.name] = graph.add_constant(initializer.name, array)

        for value_info in model.graph.input:
            if value_info.name in env:
                continue
            env[value_info.name] = graph.add_input(
                value_info.name,
                self._tensor_type_from_value_info(value_info),
            )

        for node in model.graph.node:
            op_type = self._op_map.get(node.op_type)
            if op_type is None:
                raise NotImplementedError(f"Unsupported ONNX op: {node.op_type}")
            inputs = [env[name] for name in node.input if name]
            outputs = [graph.new_value(name) for name in node.output]
            attrs = {attribute.name: self._decode_attribute(attribute) for attribute in node.attribute}
            graph.add_node(Node(node.name or node.op_type.lower(), op_type, inputs, outputs, attrs))
            for output in outputs:
                env[output.name] = output

        graph.outputs = [env[item.name] for item in model.graph.output]
        return graph

    def _tensor_type_from_value_info(self, value_info: Any) -> TensorType:
        tensor_type = value_info.type.tensor_type
        shape = []
        for dim in tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.HasField("dim_value") else -1)
        dtype = str(tensor_type.elem_type)
        return TensorType(shape=tuple(shape), dtype=dtype)

    def _decode_attribute(self, attribute: Any) -> Any:
        if attribute.type == onnx.AttributeProto.INT:
            return attribute.i
        if attribute.type == onnx.AttributeProto.FLOAT:
            return attribute.f
        if attribute.type == onnx.AttributeProto.STRING:
            return attribute.s.decode("utf-8")
        if attribute.type == onnx.AttributeProto.INTS:
            return list(attribute.ints)
        if attribute.type == onnx.AttributeProto.FLOATS:
            return list(attribute.floats)
        return np.array([])
