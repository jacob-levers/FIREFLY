#!/usr/bin/env python3
"""Print the source-install dependency fingerprint used by FIREFLY launchers.

This intentionally uses only the standard library: it must be executable by a
newly-created virtual environment before FIREFLY's dependencies are installed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import sysconfig


def python_abi() -> str:
    """Interpreter/extension ABI identity, not merely ``major.minor``."""
    return "|".join((
        getattr(sys.implementation, "name", ""),
        getattr(sys.implementation, "cache_tag", "") or "",
        sysconfig.get_config_var("SOABI") or "",
    ))


def fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"firefly-source-dependencies-v1\0")
    digest.update(f"python={sys.version_info.major}.{sys.version_info.minor}\0".encode())
    digest.update(f"abi={python_abi()}\0".encode())
    digest.update((root / "pyproject.toml").read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    print(fingerprint(Path(__file__).resolve().parents[1]))
