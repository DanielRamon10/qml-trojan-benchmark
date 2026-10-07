"""End-to-end smoke test: the whole quick pipeline runs and produces expected artifacts.

Marked ``slow`` (a few minutes). Run with ``pytest -m slow`` or the full suite.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
STAGES = ["exp1_golden.py", "exp2_trojans.py", "exp3_backdoor.py", "exp4_detection.py"]


@pytest.mark.slow
def test_quick_pipeline_end_to_end(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    for stage in STAGES:
        monkeypatch.setattr(sys, "argv", [stage, "--config", "quick"])
        runpy.run_path(str(SCRIPTS / stage), run_name="__main__")

    results = REPO / "results" / "quick"
    for name in ("golden_accuracy.csv", "metadata.csv", "sensitivity.csv", "backdoor.csv",
                 "detection_summary.csv", "inventory.csv"):
        assert (results / name).exists(), f"missing {name}"
    assert list((results / "golden").glob("*.qasm"))
    assert list((results / "infected").glob("*.qasm"))
    assert list((results / "figures").glob("*.png"))
