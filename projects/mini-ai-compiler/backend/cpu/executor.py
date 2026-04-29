from __future__ import annotations

from typing import Any

import numpy as np

from ir.graph import Graph


class CPUExecutor:
    def run(self, graph: Graph, inputs: dict[str, Any]) -> list[np.ndarray]:
        env: dict[str, np.ndarray] = {}

        for value in graph.inputs:
            env[value.name] = np.asarray(inputs[value.name])

        for node in graph.nodes:
            if node.op_type == "constant":
                env[node.outputs[0].name] = np.asarray(node.attrs["value"])
                continue

            args = [env[input_value.name] for input_value in node.inputs]

            if node.op_type == "add":
                result = args[0] + args[1]
            elif node.op_type == "mul":
                result = args[0] * args[1]
            elif node.op_type == "matmul":
                result = args[0] @ args[1]
            elif node.op_type == "relu":
                result = np.maximum(args[0], 0)
            elif node.op_type == "linear":
                result = args[0] @ args[1].T
                if len(args) == 3:
                    result = result + args[2]
            elif node.op_type == "fused_linear_relu":
                result = args[0] @ args[1].T
                if len(args) == 3:
                    result = result + args[2]
                result = np.maximum(result, 0)
            else:
                raise NotImplementedError(f"Unsupported CPU op: {node.op_type}")

            env[node.outputs[0].name] = np.asarray(result)

        return [env[value.name] for value in graph.outputs]
