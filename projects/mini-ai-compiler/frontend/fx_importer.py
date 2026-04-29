from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.fx as fx
except ImportError:  # pragma: no cover
    torch = None
    fx = None

from ir.graph import Graph
from ir.node import Node
from ir.types import TensorType
from ir.value import Value


@dataclass
class ImportedValue:
    value: Value


class FXImporter:
    def import_model(self, model: "torch.nn.Module", example_inputs: tuple[Any, ...]) -> Graph:
        if torch is None or fx is None:
            raise RuntimeError("PyTorch is required for FX import.")

        traced = fx.symbolic_trace(model)
        graph = Graph(name=type(model).__name__)
        env: dict[str, Value] = {}

        modules = dict(traced.named_modules())
        params = dict(traced.named_parameters())
        buffers = dict(traced.named_buffers())

        for node in traced.graph.nodes:
            if node.op == "placeholder":
                example = example_inputs[len(graph.inputs)]
                value = graph.add_input(
                    node.name,
                    self._tensor_type_from_runtime(example),
                )
                env[node.name] = value
                continue

            if node.op == "output":
                outputs = node.args[0]
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                graph.outputs = [env[item.name] for item in outputs]
                continue

            if node.op == "call_module":
                module = modules[node.target]
                if isinstance(module, torch.nn.Linear):
                    input_value = env[node.args[0].name]
                    weight_value = graph.add_constant(
                        f"{node.name}_weight",
                        module.weight.detach().cpu().numpy(),
                    )
                    inputs = [input_value, weight_value]
                    if module.bias is not None:
                        bias_value = graph.add_constant(
                            f"{node.name}_bias",
                            module.bias.detach().cpu().numpy(),
                        )
                        inputs.append(bias_value)
                    output = graph.new_value(node.name)
                    graph.add_node(
                        Node(
                            name=node.name,
                            op_type="linear",
                            inputs=inputs,
                            outputs=[output],
                        )
                    )
                    env[node.name] = output
                    continue

                if isinstance(module, torch.nn.ReLU):
                    input_value = env[node.args[0].name]
                    output = graph.new_value(node.name)
                    graph.add_node(Node(node.name, "relu", [input_value], [output]))
                    env[node.name] = output
                    continue

                raise NotImplementedError(f"Unsupported module call: {type(module).__name__}")

            if node.op == "call_function":
                mapped = self._map_function_target(node.target)
                if mapped is None:
                    raise NotImplementedError(f"Unsupported function call: {node.target}")
                inputs = [self._resolve_arg(arg, env, graph) for arg in node.args]
                output = graph.new_value(node.name)
                graph.add_node(Node(node.name, mapped, inputs, [output]))
                env[node.name] = output
                continue

            if node.op == "get_attr":
                target = node.target
                tensor = None
                if target in params:
                    tensor = params[target].detach().cpu().numpy()
                elif target in buffers:
                    tensor = buffers[target].detach().cpu().numpy()
                else:
                    attr = getattr(traced, target)
                    if hasattr(attr, "detach"):
                        tensor = attr.detach().cpu().numpy()
                    else:
                        tensor = np.array(attr)
                env[node.name] = graph.add_constant(node.name, tensor)
                continue

            raise NotImplementedError(f"Unsupported FX node op: {node.op}")

        return graph

    def _resolve_arg(self, arg: Any, env: dict[str, Value], graph: Graph) -> Value:
        if hasattr(arg, "name") and arg.name in env:
            return env[arg.name]
        if isinstance(arg, (int, float)):
            return graph.add_constant(f"const_{len(graph.nodes)}", np.array(arg))
        raise NotImplementedError(f"Unsupported FX argument: {arg!r}")

    def _map_function_target(self, target: Any) -> str | None:
        if target in (operator.add, torch.add if torch else object()):
            return "add"
        if target in (operator.mul, torch.mul if torch else object()):
            return "mul"
        if target in (torch.matmul if torch else object(),):
            return "matmul"
        return None

    def _tensor_type_from_runtime(self, value: Any) -> TensorType:
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            shape = tuple(int(dim) for dim in value.shape)
            dtype = str(value.dtype).replace("torch.", "")
            return TensorType(shape=shape, dtype=dtype)
        array = np.asarray(value)
        return TensorType(shape=array.shape, dtype=str(array.dtype))
