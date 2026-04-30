from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    from tools import export_bridge_mlir


@unittest.skipUnless(HAS_TORCH, "torch is required for export_bridge_mlir tests")
class ExportBridgeMLIRToolTests(unittest.TestCase):
    def test_main_writes_bridge_mlir_artifact(self) -> None:
        output_path = Path(export_bridge_mlir.PROJECT_ROOT) / "artifacts" / "mlp_bridge_input.mlir"
        if output_path.exists():
            output_path.unlink()

        export_bridge_mlir.main()

        self.assertTrue(output_path.exists())
        text = output_path.read_text(encoding="utf-8")
        self.assertIn("module {", text)
        self.assertIn("func.func", text)
        self.assertIn('"mini.', text)


if __name__ == "__main__":
    unittest.main()
