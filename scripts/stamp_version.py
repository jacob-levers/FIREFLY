#!/usr/bin/env python3
"""Stamp FIREFLY's source version for a release build.

Both platform build jobs call this helper so package metadata, the bundled
source version, and macOS plist metadata all originate from the same tag.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import plistlib
import re
import sys

_SOURCE_RE = re.compile(r'^__version__ = ".*"$', re.MULTILINE)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from firefly.release import (
    normalise_release_version,
    numeric_build_version,
    release_base,
)


def normalise(version: str) -> str:
    return normalise_release_version(version)


def stamp(source: Path, version: str) -> None:
    text = source.read_text(encoding="utf-8")
    updated, count = _SOURCE_RE.subn(f'__version__ = "{version}"', text, count=1)
    if count != 1:
        raise RuntimeError(f"could not find one __version__ assignment in {source}")
    source.write_text(updated, encoding="utf-8")


def _source_bundle_build_version(version: str) -> str:
    """Derive a numeric build value for the tracked source-launcher bundle."""
    value = normalise(version)
    suffix = value.split("-", 1)[1] if "-" in value else ""
    match = re.search(r"(?:^|[.-])(\d+)$", suffix)
    candidate = match.group(1) if match else release_base(value)
    return numeric_build_version(candidate)


def stamp_plist(plist_path: Path, version: str,
                build_number: str | None = None) -> None:
    """Keep the source-launcher plist aligned with the shared release version."""
    value = normalise(version)
    with plist_path.open("rb") as handle:
        payload = plistlib.load(handle)
    payload["CFBundleShortVersionString"] = release_base(value)
    payload["CFBundleVersion"] = numeric_build_version(
        build_number, fallback=_source_bundle_build_version(value))
    payload["FIREFLYReleaseVersion"] = value
    with plist_path.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", nargs="?", default=os.environ.get("FIREFLY_BUILD_TAG"))
    parser.add_argument("--source", type=Path,
                        default=_ROOT / "firefly" / "release.py")
    parser.add_argument(
        "--plist", type=Path,
        default=_ROOT / "Launch_FIREFLY.app" / "Contents" / "Info.plist")
    parser.add_argument(
        "--build-number",
        help=("numeric CFBundleVersion for the source launcher; prereleases "
              "default to their trailing release-candidate number"))
    args = parser.parse_args()
    version = normalise(args.version or "")
    stamp(args.source, version)
    stamp_plist(args.plist, version, build_number=args.build_number)
    print(f"Stamped {args.source} and {args.plist}: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
