"""Smoke test: the benchmark harness itself must keep executing.

Real measurements are on-demand (see benchmarks/README.md); this only
proves the harness runs and writes a complete report on a tiny problem.
"""

import json
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).parent.parent / "benchmarks" / "run.py"


def test_harness_writes_complete_report(tmp_path):
    out = tmp_path / "bench.json"
    subprocess.run(
        [sys.executable, str(RUNNER), "--scale", "smoke", "--out", str(out)],
        check=True,
        capture_output=True,
    )
    report = json.loads(out.read_text())
    env = report["environment"]
    for field in ("geodef", "numpy", "backend", "precision", "machine"):
        assert env[field]
    (result,) = report["results"]
    assert result["scale"] == "smoke"
    for stage in (
        "greens_assembly_cold",
        "greens_assembly_cached",
        "system_preparation",
        "regularized_solve",
    ):
        assert result["stages"][stage]["seconds"] >= 0
        assert result["stages"][stage]["peak_bytes"] > 0
