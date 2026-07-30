"""Read FIREFLY's release version without importing the analysis core."""
from __future__ import annotations

from firefly.release import release_version


def app_version() -> str:
    """Complete package/update version, including any prerelease suffix."""
    return release_version()
