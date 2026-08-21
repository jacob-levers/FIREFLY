"""FIREFLY release metadata shared by packages, the UI, and frozen builds.

This module is intentionally standard-library-only so build tooling can import
it without importing Qt, matplotlib, torch, or the analysis pipeline.
"""
from __future__ import annotations

import re


__version__ = "2.76.50-rc.4"

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
_NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)*$")


def normalise_release_version(version: str) -> str:
    """Validate and return a tag/package version without a leading ``v``."""
    value = (version or "").strip().lstrip("vV")
    if not _VERSION_RE.fullmatch(value):
        raise ValueError(f"invalid release version: {version!r}")
    return value


def release_base(version: str = __version__) -> str:
    """Return the numeric marketing-version core (for example ``2.76.45``)."""
    value = normalise_release_version(version)
    return value.split("-", 1)[0]


def numeric_build_version(value: str | None, *,
                          fallback: str | None = None) -> str:
    """Validate Apple's numeric/dotted ``CFBundleVersion`` value."""
    candidate = str(value or fallback or release_base()).strip()
    if not _NUMERIC_RE.fullmatch(candidate):
        raise ValueError(
            f"FIREFLY build number must be numeric/dotted, got {candidate!r}")
    return candidate


def release_version() -> str:
    """Return the complete package/update version, including prerelease suffix."""
    return __version__
