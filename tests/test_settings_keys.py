"""Every settings key the code reads must be one the app can actually write.

A key is a bare string.  Read one that nothing writes and you get the default
forever, with no error: the setting simply stops applying in that one place.
The Log-D clip range was migrated to `analysis/dcoeff_clip_logmin` / `logmax`,
runs honoured it, and the Analysis tab went on reading the retired
`dcoeff_clip_min` / `max` — so the live comparison clamped at 1e-5…10 µm²/s
whatever the sidebar said, and never redrew when you changed it.
"""
import collections
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1] / "firefly"

# Keys that are real but are not sidebar parameters: app state and preferences,
# each written by its own controller or dialog.
NOT_SIDEBAR = {
    "analysis/file":              "last input file — ImportController",
    "analysis/outdir":            "output folder — ImportController",
    "analysis/output_explicit":   "whether the output folder was chosen — ImportController",
    "analysis/metric":            "selected Analysis-tab metric — workspace",
    "analysis/presets":           "saved Analysis-tab presets — workspace",
    "analysis/roi_sister_suffix": "companion-image suffix — Preferences dialog",
}

_GETTER = re.compile(
    r'\b(?:get_str|get_float|get_bool|get_int|get|getStr|getBool|setValue|set)'
    r'\(\s*["\']((?:analysis|imaging|roi|drift|cluster|perf)/[\w.]+)["\']')


def _schema_keys():
    from firefly.ui.controllers.params.sidebar_schema import FIELDS
    keys = set()
    for f in FIELDS:
        for k in ("key", "key2"):
            if isinstance(f.get(k), str):
                keys.add(f[k])
    return keys


def _reads():
    out = collections.defaultdict(list)
    for f in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.qml")):
        if f.name == "sidebar_schema.py":
            continue
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for k in _GETTER.findall(line):
                out[k].append(f"{f.relative_to(ROOT.parent)}:{ln}")
    return out


def test_every_settings_key_the_code_uses_is_defined():
    defined = _schema_keys() | set(NOT_SIDEBAR)
    orphans = {k: locs for k, locs in _reads().items() if k not in defined}
    assert not orphans, (
        "settings keys read or written that nothing defines — each silently "
        f"falls back to its default: {dict(orphans)}")


def test_the_allowlist_has_not_gone_stale():
    """An allowlisted key nobody uses any more should come off the list, or it
    would hide the next typo that happens to match it."""
    used = set(_reads())
    stale = sorted(k for k in NOT_SIDEBAR if k not in used)
    assert not stale, f"no longer used; remove from NOT_SIDEBAR: {stale}"


def test_presets_only_set_keys_the_schema_knows():
    """A preset key the schema doesn't know is dropped on load, silently."""
    defined = _schema_keys()
    for pf in sorted((ROOT / "ui" / "presets").glob("*.json")):
        keys = [k for k in json.loads(pf.read_text()) if "/" in k]
        unknown = sorted(k for k in keys if k not in defined)
        assert not unknown, f"{pf.name} sets keys the schema does not define: {unknown}"
