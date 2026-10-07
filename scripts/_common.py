"""Shared helpers for the experiment scripts: config loading, paths and logging."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
RESULTS = REPO_ROOT / "results"


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout,
    )
    # Qiskit's transpiler/passmanager emit very verbose INFO logs; keep them quiet.
    for noisy in ("qiskit", "qiskit.passmanager", "qiskit.compiler", "qiskit.transpiler",
                  "stevedore", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


@dataclass
class Paths:
    """Resolved output directories for one run mode."""

    mode: str
    root: Path

    @property
    def golden(self) -> Path:
        return self.root / "golden"

    @property
    def infected(self) -> Path:
        return self.root / "infected"

    @property
    def backdoor(self) -> Path:
        return self.root / "backdoor"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    def ensure(self) -> Paths:
        for d in (self.root, self.golden, self.infected, self.backdoor, self.figures):
            d.mkdir(parents=True, exist_ok=True)
        return self


def load_config(mode: str) -> dict:
    """Load ``configs/<mode>.yaml`` (``quick`` or ``full``)."""
    path = CONFIGS / f"{mode}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_paths(mode: str) -> Paths:
    """Quick mode writes to results/quick (gitignored); full to results/full (versioned)."""
    return Paths(mode=mode, root=RESULTS / mode).ensure()


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true",
                   help="Use the quick config (minutes) instead of the full benchmark.")
    p.add_argument("--config", default=None, help="Explicit config name (overrides --quick).")
    return p.parse_args()


def mode_from_args(args: argparse.Namespace) -> str:
    if args.config:
        return args.config
    return "quick" if args.quick else "full"


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Timer:
    """Context manager that logs the wall-clock duration of a block."""

    def __init__(self, logger: logging.Logger, label: str):
        self.logger = logger
        self.label = label

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.logger.info("%s took %.1f s", self.label, time.perf_counter() - self.start)
