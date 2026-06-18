"""Phase-7 launch smoke: run_firefly.py boots each front-end to a rendered root.

Exercises the real entry path (selector → build_main_window → QML scene-graph
paint → SPTPALM_READY_MARKER), the non-frozen equivalent of the packaging smoke.
A "blank window" (e.g. a missing Qt Quick plugin once frozen, or a QML load
error) never writes the marker, so this fails. Qt-gated + offscreen; skipped in
the Qt-less CI image.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import subprocess                                        # noqa: E402
import sys                                               # noqa: E402
import time                                              # noqa: E402
import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("PySide6.QtQuickWidgets")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _boot(ui_env, marker, timeout=40):
    env = dict(os.environ)
    env["SPTPALM_READY_MARKER"] = str(marker)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["QT_QUICK_BACKEND"] = "software"
    if ui_env is not None:
        env["FIREFLY_UI"] = ui_env
    proc = subprocess.Popen([sys.executable, os.path.join(_ROOT, "run_firefly.py")],
                            cwd=_ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.exists(marker):
                return True
            if proc.poll() is not None:           # exited before the marker
                return os.path.exists(marker)
            time.sleep(0.25)
        return False
    finally:
        proc.terminate()
        try:    proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_qml_frontend_boots_to_ready(tmp_path):
    marker = tmp_path / "ready_qml"
    assert _boot("qml", marker), "QML front-end never reached the ready marker"


def test_widgets_frontend_still_boots(tmp_path):
    marker = tmp_path / "ready_widgets"
    assert _boot(None, marker), "Widgets front-end never reached the ready marker"
