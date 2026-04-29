from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    from tools import export_mlir


@unittest.skipUnless(HAS_TORCH, "torch is required for export_mlir tool tests")
class ExportMLIRToolTests(unittest.TestCase):
    def test_main_writes_mlir_artifact(self) -> None:
        output_path = Path(export_mlir.PROJECT_ROOT) / "artifacts" / "mlp_optimized.mlir"
        if output_path.exists():
            output_path.unlink()

        export_mlir.main()

        self.assertTrue(output_path.exists())
        text = output_path.read_text(encoding="utf-8")
        self.assertIn("module {", text)
        self.assertIn("func.func", text)
        self.assertIn("mini.", text)


if __name__ == "__main__":
    unittest.main()
