"""Run the full benchmark pipeline: experiments 1 -> 4 in order.

Usage:
    python scripts/run_all.py --quick     # minutes, validates the whole pipeline
    python scripts/run_all.py             # full benchmark (~40-50 min on a laptop)
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from _common import get_logger, mode_from_args, parse_args

LOG = get_logger("run_all")
SCRIPTS = Path(__file__).resolve().parent
STAGES = ["exp1_golden.py", "exp2_trojans.py", "exp3_backdoor.py", "exp4_detection.py"]


def main() -> None:
    args = parse_args(__doc__)
    mode = mode_from_args(args)
    forward = ["--config", mode]
    LOG.info("Running pipeline in '%s' mode", mode)
    for stage in STAGES:
        LOG.info("=== %s ===", stage)
        sys.argv = [stage, *forward]
        runpy.run_path(str(SCRIPTS / stage), run_name="__main__")
    LOG.info("Pipeline complete (mode=%s).", mode)


if __name__ == "__main__":
    main()
