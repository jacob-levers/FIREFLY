"""Names that resolve to nothing at runtime, inside handlers that swallow.

Two live defects were found this way, both in ``_run_one_analysis`` and both
invisible because the failing expression sat inside ``except Exception``:

  * ``np.hypot(...)`` — this module imports numpy as ``_np`` and binds ``np``
    nowhere, so the drift-extent QC number raised NameError on every run and was
    replaced with None.  FIREFLY never reported sample drift.
  * ``proj_sample.shape`` — read several hundred lines AFTER ``del
    proj_sample``, so the super-resolution renderer never received the camera
    field size and silently fell back to the localisations' bounding box.  Every
    exported reconstruction was cropped to wherever molecules happened to be.

Neither raised, neither logged, and neither changed a number that anyone was
checking — which is precisely why they survived.  81% of this package's
exception handlers swallow without logging, so a name that resolves to nothing
degrades output rather than failing loudly.  These checks are static because the
code paths need a real movie to execute; the point is to catch the class without
needing one.
"""
import ast
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "firefly")


def _py_files():
    for dirpath, _dirnames, files in os.walk(PKG):
        if "__pycache__" in dirpath:
            continue
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def _rel(path):
    return os.path.relpath(path, ROOT)


# ── a name read after it has been deleted ────────────────────────────────────
def _read_after_del(path):
    """(function, name, del_line, read_line) for every name read after a ``del``
    with no rebinding in between.  A read BEFORE the del is fine (that is the
    normal case, including a loop that rebinds at the top of each pass)."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        dels, binds, reads = {}, {}, {}
        for n in ast.walk(fn):
            if isinstance(n, ast.Name):
                if isinstance(n.ctx, ast.Del):
                    dels.setdefault(n.id, []).append(n.lineno)
                elif isinstance(n.ctx, ast.Store):
                    binds.setdefault(n.id, []).append(n.lineno)
                else:
                    reads.setdefault(n.id, []).append(n.lineno)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    binds.setdefault(
                        a.asname or a.name.split(".")[0], []).append(n.lineno)
        for name, del_lines in dels.items():
            for d in del_lines:
                for r in reads.get(name, []):
                    if r <= d:
                        continue
                    if any(d < b < r for b in binds.get(name, [])):
                        continue
                    out.append((fn.name, name, d, r))
    return out


def test_no_name_is_read_after_it_is_deleted():
    """``del`` here is a memory optimisation — a 16k-frame stack is freed the
    moment the figure is drawn.  Reading the name afterwards is a guaranteed
    NameError, and the surrounding handler turns it into a quiet wrong answer
    rather than a crash."""
    offenders = []
    for p in _py_files():
        for fnname, name, d, r in _read_after_del(p):
            offenders.append(f"{_rel(p)}:{r}: `{name}` read after `del` "
                             f"at line {d} (in {fnname}())")
    assert not offenders, "\n".join(offenders)


# ── the numpy alias this module actually binds ───────────────────────────────
def test_the_worker_uses_the_numpy_alias_it_imports():
    """firefly_worker imports numpy as ``_np`` (per-function, to keep startup
    off matplotlib/numpy).  A bare ``np.`` there resolves to nothing."""
    path = os.path.join(PKG, "firefly_worker.py")
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    bound = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                bound.add(a.asname or a.name.split(".")[0])
    assert "_np" in bound, "the worker no longer imports numpy as _np"

    lines = src.splitlines()
    bad = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == "np" and "np" not in bound):
            bad.append(f"firefly/firefly_worker.py:{n.lineno}: "
                       f"{lines[n.lineno - 1].strip()[:90]}")
    assert not bad, ("bare `np.` in a module that binds numpy as `_np`:\n"
                     + "\n".join(bad))


# ── the contract the super-resolution fix relies on ──────────────────────────
def test_render_superres_field_px_controls_the_canvas():
    """The fix hands ``render_superres`` the real field size instead of letting
    it fall back.  Pin both halves of that contract: given a field the canvas is
    the FIELD, and given None it is the bounding box — the documented fallback
    that the deleted-name bug was silently taking on every run."""
    import numpy as np
    from firefly.analysis.fa_render import render_superres

    rng = np.random.default_rng(0)
    # molecules confined to one corner of a 512x512 camera field
    x = rng.normal(300.0, 20.0, 3000)
    y = rng.normal(180.0, 20.0, 3000)

    full = render_superres(x, y, 0.1, sr_nm=20.0, blur_nm=20.0, field_px=(512, 512))
    bbox = render_superres(x, y, 0.1, sr_nm=20.0, blur_nm=20.0, field_px=None)

    assert full.shape[0] > bbox.shape[0] and full.shape[1] > bbox.shape[1], (
        "field_px is not controlling the canvas, so the worker fix is inert")
    # the full-field canvas is the camera field at the super-res pixel size
    assert full.shape == (512 * 100 // 20, 512 * 100 // 20)


def test_the_worker_passes_the_field_size_it_recorded(monkeypatch):
    """stack_h/stack_w are the field dimensions already written to the CSV
    header and the run manifest.  The super-resolution canvas must use those
    same numbers, so the reconstruction lines up with the trajectory and density
    panels instead of being cropped to the molecules."""
    src = open(os.path.join(PKG, "firefly_worker.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_run_one_analysis")

    # find the assignment to _field and confirm it reads stack_h / stack_w
    field_srcs = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_field" for t in n.targets):
            names = {m.id for m in ast.walk(n.value) if isinstance(m, ast.Name)}
            field_srcs.append(names)
    assert field_srcs, "_field is no longer assigned in _run_one_analysis"
    assert any({"stack_h", "stack_w"} <= names for names in field_srcs), (
        f"_field is not built from the recorded field size: {field_srcs}")
    assert not any("proj_sample" in names for names in field_srcs), (
        "_field reads proj_sample again — it is deleted before this point")
