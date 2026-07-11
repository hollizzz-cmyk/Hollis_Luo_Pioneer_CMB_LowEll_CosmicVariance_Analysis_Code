"""Compatibility wrapper for the project's src/cutoff_models.py module."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SRC_FILE = Path(__file__).resolve().parents[1] / "src" / "cutoff_models.py"
SPEC = importlib.util.spec_from_file_location("_pioneer_src_cutoff_models", SRC_FILE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load {SRC_FILE}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

for name in dir(MODULE):
    if not name.startswith("_"):
        globals()[name] = getattr(MODULE, name)
