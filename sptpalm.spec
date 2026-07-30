# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for FIREFLY (PySide6 frontend; bespoke Qt viewers, no napari).

Build (from the project root):
  macOS:   pyinstaller sptpalm.spec
  Windows: pyinstaller sptpalm.spec

Outputs:
  macOS:   dist/FIREFLY.app  (then wrap in a DMG via CI)
  Windows: dist/FIREFLY.exe  (onefile)
"""

from PyInstaller.utils.hooks import (
    collect_submodules, collect_data_files, copy_metadata)
import os
import sys

from firefly.release import (
    __version__ as _FULL_VERSION,
    numeric_build_version,
    release_base,
)


def _collect_no_tests(pkg):
    """collect_submodules(pkg) MINUS test subpackages.

    The scientific wheels ship large ``*.tests.*`` trees (pandas alone ~1,100
    modules; ~2,300 across pandas/scipy/sklearn/statsmodels/numpy) that
    PyInstaller would otherwise analyse AND bundle into the onefile — code never
    imported at run time, bloating both the .exe and the build.  Drops only the
    ``tests`` package; ``testing`` utilities (e.g. numpy.testing, which some
    libraries import at run time) are deliberately KEPT.
    """
    return collect_submodules(pkg, filter=lambda n: "tests" not in n.split("."))


# Apple's marketing version is the numeric release core. CI supplies its
# numeric run number for CFBundleVersion; local builds fall back to the release
# core. Keep the exact tag (including an rc suffix) in FIREFLY's dedicated key.
_BUNDLE_VERSION = release_base(_FULL_VERSION)
_CI_BUILD_VERSION = numeric_build_version(
    os.environ.get("FIREFLY_BUILD_NUMBER"), fallback=_BUNDLE_VERSION)


# ── Hidden imports ───────────────────────────────────────────────────────────
hidden = []

# Scientific Python — collect every submodule because lazy imports under
# numpy._core / pandas._libs / scipy.* are missed by static analysis.  No-tests
# variant: the wheels' bundled `*.tests.*` trees are never imported at run time.
hidden += _collect_no_tests("numpy")
hidden += _collect_no_tests("pandas")
hidden += _collect_no_tests("trackpy")
hidden += _collect_no_tests("scipy")
hidden += _collect_no_tests("skimage")
hidden += _collect_no_tests("sklearn")
hidden += _collect_no_tests("matplotlib")
hidden += _collect_no_tests("joblib")
hidden += _collect_no_tests("aicspylibczi")
hidden += _collect_no_tests("imagecodecs")

# Vendored sub-packages under scipy._external / sklearn.externals.  scipy >=1.16
# moved array_api_compat (+ array_api_extra, packaging_version) under
# `scipy._external`, and collect_submodules("scipy") does NOT recurse into it —
# so e.g. scipy._external.array_api_compat.numpy.fft (pulled in transitively by
# scipy.ndimage <- trackpy <- sptpalm_analysis) is absent from the bundle.  The
# result: the analysis WORKER dies on `import sptpalm_analysis` with
# `ModuleNotFoundError` the instant a run starts, while the GUI — which never
# imports the analysis stack (see firefly.ui._appversion) — launches fine.  That
# "app opens, every single/HYPER-FLY run fails with no visible error" is exactly
# the Windows-frozen failure this closes.  collect_submodules of the _external /
# externals PARENT is itself flaky here (misses packaging_version / _numpydoc),
# so collect each vendored subtree explicitly.
for _vendored in (
    "scipy._external.array_api_compat",
    "scipy._external.array_api_extra",
    "scipy._external._array_api_compat_vendor",
    "scipy._external.packaging_version",
    "sklearn.externals._numpydoc",
):
    try:
        hidden += collect_submodules(_vendored)
    except Exception:
        pass

# pingouin (two-way mixed ANOVA for the Compare tab) and its heavy deps.
# Optional at runtime — fa_twoway guards the import — so only collect if
# actually installed, to avoid inflating the bundle with missing stubs.
for _opt in ("pingouin", "statsmodels", "pandas_flavor", "outdated"):
    try:
        __import__(_opt)
        hidden += _collect_no_tests(_opt)
    except Exception:
        pass

# certifi — frozen builds don't ship a usable CA store, so HTTPS verification
# (auto-update check, in-app CUDA-wheel installer) fails without certifi's
# bundled cacert.pem.  Its data file is collected below.
hidden += collect_submodules("certifi")

# Qt / PySide6 — the Qt6 stack.  collect_submodules pulls in plugin loaders.
hidden += collect_submodules("PySide6")
hidden += collect_submodules("shiboken6")

# napari has been removed — the interactive viewers are bespoke Qt widgets,
# so there's no napari / vispy / magicgui / npe2 / qtpy / superqt plugin
# discovery to bundle any more.

# PyTorch — GPU localiser backend.  Only collect if the dep is installed;
# otherwise we'd inflate the bundle with non-existent stubs.
try:
    import torch  # noqa: F401
    hidden += collect_submodules("torch")
except ImportError:
    pass

# numba / llvmlite — trackpy's FAST locate engine.  trackpy imports numba
# lazily (inside its refine path), so PyInstaller's static analysis misses it;
# without these the frozen app silently falls back to the ~5-10x slower
# pure-Python locate engine, which dominates the auto-threshold harvest.
# PyInstaller's bundled numba/llvmlite hooks pull in the LLVM shared library.
try:
    import numba  # noqa: F401
    hidden += collect_submodules("numba")
    hidden += collect_submodules("llvmlite")
except ImportError:
    pass

hidden += [
    "czifile", "aicspylibczi", "imagecodecs",
    "tifffile",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.backend_agg",
    "PIL._tkinter_finder", "PIL.Image", "PIL.ImageTk",
    "PIL.ImageFile",
    "PIL.PngImagePlugin",
    "PIL.JpegImagePlugin",
    "PIL.TiffImagePlugin",
    "PIL.BmpImagePlugin",
    "PIL.GifImagePlugin",
    "PIL.WebPImagePlugin",
    "pandas._libs.tslibs.np_datetime",
    "pandas._libs.tslibs.nattype",
    "pandas._libs.tslibs.timedeltas",
    "pandas._libs.tslibs.timestamps",
    "multiprocessing.pool",
    "multiprocessing.managers",
    "concurrent.futures",
    "concurrent.futures.thread",
    "psutil",
    "threadpoolctl",
    # Encoding tables sometimes missed in frozen builds
    "encodings.utf_8",
    "encodings.ascii",
    "encodings.latin_1",
]

# FIREFLY's own package — collect every submodule (firefly, firefly.analysis.*,
# firefly.ui.*) so the spawned worker subprocess can re-import
# firefly.firefly_worker.run_analysis and the lazily-imported analysis/UI
# modules are all present in the frozen bundle.
hidden += collect_submodules("firefly")

# Qt Quick front-end (the only UI).  run_firefly.py imports app_qml LAZILY,
# so PyInstaller's static analysis can miss these — name them explicitly.  The
# PySide6 hook then pulls the Quick plugin + scenegraph trees + qmldir manifests
# (a missing one is the classic "blank frozen window").  QtSvg backs the Lucide
# icon image-provider; QtQuickControls2 is restyled to the Basic style.
hidden += ["PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets",
           "PySide6.QtQuickControls2", "PySide6.QtSvg"]

# ── Datas ─────────────────────────────────────────────────────────────────────
datas = []
datas += collect_data_files("skimage")
datas += collect_data_files("matplotlib")
datas += collect_data_files("aicspylibczi")
datas += collect_data_files("PIL")
# certifi's cacert.pem — without it, certifi.where() points at a missing file
# in the frozen bundle and HTTPS still fails.
datas += collect_data_files("certifi")

# `.dist-info` metadata for scientific packages.  Without this, pandas's
# `import_optional_dependency("numpy")` (and similar checks in
# scikit-image, scipy, etc.) raises the misleading
#   "Missing optional dependency 'numpy'"
# error at runtime in the frozen build — even though the package itself
# is bundled and importable.  `importlib.metadata.version(pkg)` resolves
# against the .dist-info directory, which PyInstaller doesn't pick up
# automatically for collect_submodules.  copy_metadata fixes that.
for _pkg in ("numpy", "pandas", "scipy", "scikit-image", "scikit-learn",
             "matplotlib",
             "tifffile", "trackpy", "joblib", "Pillow", "psutil",
             "torch",
             "PySide6", "shiboken6",
             # imageio + lazy_loader are real deps (scikit-image uses both).
             "imageio", "lazy_loader",
             # numba/llvmlite do importlib.metadata version lookups at import.
             "numba", "llvmlite",
             # pingouin (Compare-tab two-way ANOVA) + deps do runtime
             # importlib.metadata version lookups; without .dist-info the
             # frozen import raises PackageNotFoundError.
             "pingouin", "statsmodels", "pandas_flavor", "outdated"):
    try:
        datas += copy_metadata(_pkg)
    except Exception:
        # Package not installed at freeze time — fine, just skip it.
        pass

# Ship the source .py for the top-level package modules alongside the bundle.
# firefly_worker reads __file__ to locate the git SHA; crash_reporter embeds
# source context in reports.  Mirror the package path so __file__-relative
# lookups resolve.
datas += [("firefly/sptpalm_analysis.py", "firefly")]
datas += [("firefly/firefly_worker.py",   "firefly")]
datas += [("firefly/crash_reporter.py",   "firefly")]
datas += [("firefly/cuda_installer.py",   "firefly")]

# Bundle the app icon PNG so the Qt window/dock icon can be loaded
# at runtime from sys._MEIPASS/assets/icon.png in frozen mode.
if os.path.isfile(os.path.join(SPECPATH, "assets", "icon.png")):
    datas += [(os.path.join(SPECPATH, "assets", "icon.png"), "assets")]

# Bundle CHANGELOG.md so the landing's "Recent updates" timeline can read it
# from sys._MEIPASS/CHANGELOG.md in the frozen app (firefly.ui.changelog).
if os.path.isfile(os.path.join(SPECPATH, "CHANGELOG.md")):
    datas += [(os.path.join(SPECPATH, "CHANGELOG.md"), ".")]

# Bundle the real-data figure-preview panel thumbnails (shown in
# Preferences -> Figure defaults) so the preview works in the frozen app.
_preview_panels = os.path.join(SPECPATH, "firefly", "ui", "assets",
                               "preview_panels")
if os.path.isdir(_preview_panels):
    datas += [(_preview_panels,
               os.path.join("firefly", "ui", "assets", "preview_panels"))]

# Bundle the QML UI tree (Main.qml + components/ + tabs/ + assets/icons) so the
# QML front-end loads them from sys._MEIPASS in the frozen app — app_qml.py
# resolves them relative to its own __file__.
_qml_dir = os.path.join(SPECPATH, "firefly", "ui", "qml")
if os.path.isdir(_qml_dir):
    datas += [(_qml_dir, os.path.join("firefly", "ui", "qml"))]

# Bundle the shipped built-in presets (PC12 Cells, Drosophila Neurons) so every
# build seeds them into ~/.firefly/presets on first run (PresetController
# .seed_builtins resolves them from sys._MEIPASS/firefly/ui/presets).
_presets_dir = os.path.join(SPECPATH, "firefly", "ui", "presets")
if os.path.isdir(_presets_dir):
    datas += [(_presets_dir, os.path.join("firefly", "ui", "presets"))]

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ["run_firefly.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["streamlit", "tornado", "altair", "bokeh", "IPython",
              "notebook", "jupyter",
              # Tkinter no longer used after v2.0 — exclude to keep
              # the bundle from carrying the Tcl/Tk runtime
              "tkinter", "_tkinter", "tkinterdnd2"],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

_ICON_WIN = os.path.join(SPECPATH, "assets", "icon.ico") if (
    'SPECPATH' in dir() and os.path.isfile(
        os.path.join(SPECPATH, "assets", "icon.ico"))) else None
_ICON_MAC = os.path.join(SPECPATH, "assets", "icon.icns") if (
    'SPECPATH' in dir() and os.path.isfile(
        os.path.join(SPECPATH, "assets", "icon.icns"))) else None

# ── Windows: ONEFILE mode (single self-contained .exe) ───────────────────────
if sys.platform == "win32":
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name="FIREFLY",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        # Extract the onefile payload to a STABLE per-user app-data folder
        # instead of %TEMP%.  On locked-down / managed Windows profiles %TEMP%
        # is often redirected to a quota'd, aggressively-cleaned path (observed:
        # ...\AppData\Local\Temp\14\_MEIxxxx) where the ~hundreds of MB of DLLs
        # can fail to extract fully — surfacing as the bootloader error
        # "Failed to load Python DLL python3xx.dll. LoadLibrary: The specified
        # module could not be found."  %LOCALAPPDATA%\FIREFLY is where the app
        # already stores crash reports / logs / updates (proven writable on the
        # machines that hit this), so the extraction lands somewhere reliable.
        # PyInstaller expands %VAR% here at runtime on Windows.
        runtime_tmpdir="%LOCALAPPDATA%\\FIREFLY\\bundle",
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=_ICON_WIN,
    )
else:
    # macOS / Linux: ONEDIR mode (wrapped in .app/.dmg on macOS)
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="FIREFLY",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        argv_emulation=False,
        codesign_identity=None,
        entitlements_file=None,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name="FIREFLY",
    )

    if sys.platform == "darwin":
        app = BUNDLE(
            coll,
            name="FIREFLY.app",
            icon=_ICON_MAC,
            bundle_identifier="com.jacoblevers.firefly",
            info_plist={
                "CFBundleName": "FIREFLY",
                "CFBundleDisplayName": "FIREFLY — Fluorescence Inference & Reconstruction Engine",
                "CFBundleVersion": _CI_BUILD_VERSION,
                "CFBundleShortVersionString": _BUNDLE_VERSION,
                "FIREFLYReleaseVersion": _FULL_VERSION,
                "NSHighResolutionCapable": True,
                "LSMinimumSystemVersion": "11.0",
                "NSAppleEventsUsageDescription": "Required for analysis.",
                # torch's MPS backend uses Metal — leave the App Sandbox off
                # so GPU access works without prompting.  The app doesn't need
                # network or filesystem entitlements beyond what macOS grants
                # normally.
            },
        )
