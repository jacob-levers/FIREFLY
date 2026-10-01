"""One wavelet engine, and it is the palmTRACER one.

The original à trous engine (k·σ per frame on bandpassed frames, product of three
planes) agreed with palmTRACER on 28–53 % of tracks; the palmTRACER-style
reconstruction reproduces 94–99.5 % of them.  The old engine is removed, and
everything that named it — a saved sidebar choice, an old run manifest's
``"atrous"``, the verification app's backend string — now reaches the new one
rather than falling back to Auto without a word.
"""
import pytest

from firefly.ui.controllers.params import sidebar_schema as S

OLD_LABEL = "À trous wavelet — PyTorch (GPU)"
NEW_LABEL = "Wavelet — palmTRACER-style (CPU)"


def test_the_old_engine_is_gone():
    from firefly.analysis import fa_localize_backends, fa_localize
    assert not hasattr(fa_localize_backends, "AtrousWaveletBackend")
    assert "atrous" not in [c.name for c in fa_localize._BACKEND_REGISTRY]


def test_the_dropdown_offers_only_the_new_one():
    items = S.BY_KEY["analysis/backend"]["items"]
    assert NEW_LABEL in items and OLD_LABEL not in items


def test_a_saved_old_choice_runs_the_new_engine():
    from firefly.ui.controllers.params.params_builder import BACKEND_LABEL_TO_VALUE
    assert BACKEND_LABEL_TO_VALUE[OLD_LABEL] == "palmtracer"


def test_the_sidebar_shows_the_new_label_for_a_saved_old_choice():
    """Otherwise the dropdown shows a blank: the stored text is not an item."""
    pytest.importorskip("PySide6")
    from firefly.ui.controllers.params.sidebar_controller import SidebarController

    class _S(dict):
        def get_str(self, k, d=""): return dict.get(self, k, d)
        def get_float(self, k, d=0.0): return float(dict.get(self, k, d))
        def get_bool(self, k, d=False): return bool(dict.get(self, k, d))
        def set(self, k, v): self[k] = v

    sb = SidebarController(_S({"analysis/backend": OLD_LABEL}))
    assert sb.get("analysis/backend") == NEW_LABEL


def test_old_run_records_resolve_to_the_new_engine():
    from firefly.analysis.fa_enums import Backend
    from firefly.analysis.fa_localize import _resolve_backend
    from firefly.analysis.fa_localize_backends import PalmTracerWaveletBackend
    logs = []
    assert Backend.parse("atrous", log=logs.append) is Backend.PALMTRACER
    assert logs and "palmTRACER" in logs[0]
    assert isinstance(_resolve_backend("atrous"), PalmTracerWaveletBackend)
