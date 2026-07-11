"""Compatibility entry point for src/validate_cutoff_models.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
runpy.run_path(str(SRC_DIR / "validate_cutoff_models.py"), run_name="__main__")
