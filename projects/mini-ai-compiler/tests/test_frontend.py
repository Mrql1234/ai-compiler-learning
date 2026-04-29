from __future__ import annotations

import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_ONNX = importlib.util.find_spec("onnx") is not None

if HAS_TORCH:
    import torch

    from frontend.fx_importer import FXImporter

if HAS_ONNX:
    import onnx
    from onnx import TensorProto, helper

    from frontend.onnx_importer import ONNXImporter


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for FX importer tests")
class FrontendTests(unittest.TestCase):
    def test_import_linear_relu(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.relu = torch.nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.relu(self.linear(x))

        model = Model().eval()
        sample = torch.randn(2, 4)
        graph = FXImporter().import_model(model, (sample,))

        self.assertEqual(graph.inputs[0].name, "x")
        self.assertEqual([node.op_type for node in graph.nodes if node.op_type != "constant"], ["linear", "relu"])


@unittest.skipUnless(HAS_ONNX, "onnx is required for ONNX importer tests")
class ONNXFrontendTests(unittest.TestCase):
    def test_import_add_graph(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
        z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2])
        node = helper.make_node("Add", ["x", "y"], ["z"], name="add0")
        graph_proto = helper.make_graph([node], "add_graph", [x, y], [z])
        model = helper.make_model(graph_proto)

        graph = ONNXImporter().import_model(model)

        self.assertEqual([item.name for item in graph.inputs], ["x", "y"])
        self.assertEqual([item.op_type for item in graph.nodes], ["add"])


if __name__ == "__main__":
    unittest.main()
