"""Small setuptools hook for the repository-level runtime changelog.

``CHANGELOG.md`` stays at the repository root because GitHub release jobs and
contributors consume it there.  The installed UI also needs the same file, so
copy that canonical source into the ``firefly`` package while building a wheel
instead of maintaining a second, drift-prone changelog copy.
"""
from __future__ import annotations

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class _BuildPyWithChangelog(_build_py):
    """Expose the root changelog as ``firefly/CHANGELOG.md`` package data."""

    def _changelog_mapping(self) -> tuple[str, str]:
        root = Path(__file__).resolve().parent
        return (
            str(Path(self.build_lib) / "firefly" / "CHANGELOG.md"),
            str(root / "CHANGELOG.md"),
        )

    def _get_package_data_output_mapping(self):
        yield from super()._get_package_data_output_mapping()
        target, source = self._changelog_mapping()
        if Path(source).is_file():
            yield target, source

    def get_outputs(self, include_bytecode: bool = True):
        outputs = list(super().get_outputs(include_bytecode))
        target, source = self._changelog_mapping()
        if Path(source).is_file() and target not in outputs:
            outputs.append(target)
        return outputs


setup(cmdclass={"build_py": _BuildPyWithChangelog})
