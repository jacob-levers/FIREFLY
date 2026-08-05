"""Fast source-level contracts for the wheel/source-launch distribution path."""
from __future__ import annotations

import importlib.util
from importlib.resources import files
from pathlib import Path
import plistlib
import re
import subprocess
import sys


_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = _ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_entrypoint_and_runtime_assets_are_packaged():
    import firefly.__main__ as entry

    assert callable(entry.main)
    root = files("firefly")
    for relpath in (
        "ui/qml/Main.qml",
        "ui/qml/assets/icons/check.svg",
        "ui/assets/preview_panels/A.png",
        "ui/presets/PC12 Cells.json",
        "ui/controllers/app_controller.py",
        "ui/controllers/workspace/workspace_data.py",
    ):
        assert root.joinpath(relpath).is_file(), relpath


def test_wheel_build_maps_the_root_changelog_into_the_package():
    """The canonical root changelog must remain available after pip install."""
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    setup_hook = (_ROOT / "setup.py").read_text(encoding="utf-8")
    assert '"CHANGELOG.md"' in pyproject
    assert "include CHANGELOG.md" in manifest
    assert '"firefly" / "CHANGELOG.md"' in setup_hook


def test_dependency_fingerprint_is_stable_and_uses_the_venv_python():
    script = _ROOT / "scripts" / "dependency_fingerprint.py"
    first = subprocess.check_output([sys.executable, str(script)], text=True).strip()
    second = subprocess.check_output([sys.executable, str(script)], text=True).strip()
    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")
    helper = _load_script("dependency_fingerprint.py")
    abi = helper.python_abi()
    assert sys.implementation.name in abi
    assert sys.implementation.cache_tag in abi


def test_source_launchers_check_imports_and_preserve_compatible_torch():
    mac = (_ROOT / "Launch_FIREFLY.app" / "Contents" / "MacOS" / "launcher"
           ).read_text(encoding="utf-8")
    win = (_ROOT / "Launch_FIREFLY.bat").read_text(encoding="utf-8")
    for launcher in (mac, win):
        assert "runtime_import_check.py" in launcher
        assert "pip check" in launcher
        assert "pip install -e ." in launcher
        assert "pip install --upgrade -e ." not in launcher


def test_runtime_import_check_covers_gui_science_and_project():
    helper = _load_script("runtime_import_check.py")
    assert {"numpy", "trackpy", "torch", "PySide6", "firefly"} <= set(
        helper.MODULES)


def test_version_stamp_helper_updates_only_the_version_assignment(tmp_path):
    helper = _load_script("stamp_version.py")
    source = tmp_path / "version.py"
    source.write_text('# comment\n__version__ = "0.0.0"\nvalue = 1\n', encoding="utf-8")
    helper.stamp(source, helper.normalise("v2.76.45-rc.10"))
    assert source.read_text(encoding="utf-8") == (
        '# comment\n__version__ = "2.76.45-rc.10"\nvalue = 1\n')


def test_version_stamp_helper_updates_source_launcher_plist(tmp_path):
    helper = _load_script("stamp_version.py")
    plist_path = tmp_path / "Info.plist"
    with plist_path.open("wb") as handle:
        plistlib.dump({
            "CFBundleIdentifier": "com.jacoblevers.firefly.launcher",
            "CFBundleVersion": "1",
            "CFBundleShortVersionString": "1.0.0",
        }, handle)

    helper.stamp_plist(plist_path, helper.normalise("v2.76.45-rc.10"))

    with plist_path.open("rb") as handle:
        payload = plistlib.load(handle)
    assert payload["CFBundleShortVersionString"] == "2.76.45"
    assert payload["CFBundleVersion"] == "10"
    assert payload["FIREFLYReleaseVersion"] == "2.76.45-rc.10"


def test_macos_bundle_uses_numeric_ci_build_and_full_release_metadata():
    from firefly import release

    stamp_helper = _load_script("stamp_version.py")
    spec = (_ROOT / "sptpalm.spec").read_text(encoding="utf-8")
    workflow = (_ROOT / ".github" / "workflows" / "build.yml"
                ).read_text(encoding="utf-8")
    with (_ROOT / "Launch_FIREFLY.app" / "Contents" / "Info.plist"
          ).open("rb") as handle:
        source_launcher_plist = plistlib.load(handle)
    assert '"CFBundleShortVersionString": _BUNDLE_VERSION' in spec
    assert '"CFBundleVersion": _CI_BUILD_VERSION' in spec
    assert '"FIREFLYReleaseVersion": _FULL_VERSION' in spec
    assert "FIREFLY_BUILD_NUMBER: ${{ github.run_number }}" in workflow
    assert source_launcher_plist["CFBundleShortVersionString"] == release.release_base()
    assert source_launcher_plist["CFBundleVersion"] == (
        stamp_helper._source_bundle_build_version(release.__version__))
    assert source_launcher_plist["FIREFLYReleaseVersion"] == release.__version__


def test_release_gate_uses_clean_wheel_startup_and_exact_tag_contract():
    workflow = (_ROOT / ".github" / "workflows" / "build.yml"
                ).read_text(encoding="utf-8")
    assert "Validate tag, shared version, and changelog agreement" in workflow
    assert "from firefly.release import __version__" in workflow
    assert 'heading="## v${expected} — "' in workflow
    assert "python -m venv --system-site-packages" not in workflow
    assert '"$venv/bin/python" -m pip check' in workflow
    assert "Start the clean-installed wheel outside the checkout" in workflow
    assert '"CHANGELOG.md"' in workflow
    assert "installed wheel has no recent updates" in workflow
    assert '"$RUNNER_TEMP/firefly-release-wheel/bin/python" -m firefly' in workflow
    windows_job = workflow.split("build-windows:", 1)[1]
    assert 'https://download.pytorch.org/whl/cpu "torch>=2.6,<3"' in windows_job


def test_shared_release_helper_drives_package_ui_and_analysis_versions():
    from firefly import release
    from firefly.sptpalm_analysis import __version__ as analysis_version
    from firefly.ui._appversion import app_version

    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    spec = (_ROOT / "sptpalm.spec").read_text(encoding="utf-8")
    assert analysis_version == app_version() == release.__version__
    assert 'version = { attr = "firefly.release.__version__" }' in pyproject
    assert "from firefly.release import (" in spec
    # Assert the BEHAVIOUR (strip any prerelease suffix), not a pinned literal:
    # hardcoding the current version made every release edit this test while
    # protecting nothing, since the value is whatever release.py says.
    assert release.release_base() == release.__version__.split("-", 1)[0]
    assert re.fullmatch(r"\d+\.\d+\.\d+", release.release_base())
    assert release.release_base("9.9.9-rc.3") == "9.9.9"
    assert release.numeric_build_version("1234") == "1234"
