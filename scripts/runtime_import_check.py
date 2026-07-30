#!/usr/bin/env python3
"""Verify that a FIREFLY source environment can import its runtime stack."""
from __future__ import annotations

import importlib
import sys


MODULES = (
    "numpy", "pandas", "scipy", "matplotlib", "skimage", "trackpy", "numba",
    "joblib", "tqdm", "czifile", "imagecodecs", "aicspylibczi", "tifffile",
    "imageio", "PIL", "psutil", "sklearn", "pingouin", "threadpoolctl",
    "torch", "roifile", "PySide6", "firefly",
)


def missing_imports() -> list[str]:
    missing = []
    for module in MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:
            missing.append(f"{module}: {type(exc).__name__}: {exc}")
    return missing


def main() -> int:
    missing = missing_imports()
    if missing:
        print("FIREFLY runtime import check failed:", file=sys.stderr)
        for failure in missing:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
