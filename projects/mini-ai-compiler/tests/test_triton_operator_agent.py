from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "compiler-mlir" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from triton_operator_agent_lib import (  # noqa: E402
    AgentRunOptions,
    DEFAULT_AGENT_MEMORY,
    analyze_ncu_report,
    load_operator_spec,
    run_agent,
)


class TritonOperatorAgentTests(unittest.TestCase):
    def test_layernorm_spec_generates_plan(self) -> None:
        spec = load_operator_spec(
            Path("compiler-mlir/perf/specs/triton_agent_layernorm_a10.json")
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "layernorm_plan"
            summary = run_agent(
                spec,
                AgentRunOptions(
                    mode="plan",
                    dry_run=True,
                    emit_nvtx=False,
                    device_index=0,
                    run_dir=run_dir,
                    memory_path=Path(tmp_dir) / "memory.json",
                    max_candidates=4,
                    max_iterations=1,
                ),
            )
            self.assertEqual(summary["spec"]["operation"], "layernorm")
            self.assertTrue((run_dir / "plan.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "report.md").exists())
            self.assertEqual(len(summary["iterations"][0]["results"]), 4)
            self.assertEqual(summary["iterations"][0]["results"][0]["status"], "planned")

    def test_fused_linear_relu_dry_run_writes_commands(self) -> None:
        spec = load_operator_spec(
            Path("compiler-mlir/perf/specs/triton_agent_fused_linear_relu_a10.json")
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "linear_relu_tune"
            summary = run_agent(
                spec,
                AgentRunOptions(
                    mode="tune",
                    dry_run=True,
                    emit_nvtx=False,
                    device_index=0,
                    run_dir=run_dir,
                    memory_path=Path(tmp_dir) / "memory.json",
                    max_candidates=2,
                    max_iterations=1,
                    warmup=1,
                    repeat=2,
                ),
            )
            result = summary["iterations"][0]["results"][0]
            self.assertEqual(result["status"], "dry_run")
            self.assertIn("triton_linear_relu_bench.py", result["command_text"])
            self.assertTrue((run_dir / "summary.json").exists())

    def test_ncu_analysis_returns_resource_limited_diagnosis(self) -> None:
        spec = load_operator_spec(
            Path("compiler-mlir/perf/specs/triton_agent_matmul_a10.json")
        )
        fake_ncu = "\n".join(
            [
                "Achieved Occupancy % 31.2",
                "Compute (SM) Throughput % 42.7",
                "Memory Throughput Gbyte/s 180.5",
                "L2 Hit Rate % 82.4",
                "No Eligible % 18.1",
                "Eligible Warps Per Scheduler warp 1.2",
                "Registers Per Thread register/thread 128",
                "Dynamic Shared Memory Per Block Kbyte/block 48.0",
                "This kernel results in 2 bank conflicts",
                "which represent 8.0% of the overall",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            ncu_path = Path(tmp_dir) / "sample_ncu.txt"
            ncu_path.write_text(fake_ncu, encoding="utf-8")
            diagnosis = analyze_ncu_report(
                spec,
                ncu_path,
                best_config={
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "BLOCK_K": 32,
                    "GROUP_M": 4,
                    "num_warps": 8,
                    "num_stages": 3,
                },
            )
            self.assertEqual(diagnosis["bottleneck"], "resource-limited")
            self.assertTrue(diagnosis["next_experiments"])
            self.assertIn("num_warps", diagnosis["next_experiments"][0]["changes"])


if __name__ == "__main__":
    unittest.main()
