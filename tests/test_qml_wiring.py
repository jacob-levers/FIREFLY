"""Two ways QML fails SILENTLY, both hit while shipping the post-hoc ROI editor.

1. A controller registered only as a context property has no other Python
   reference, so it is collected as soon as ``build_main_window`` returns.  QML
   then sees ``null``, and every binding on it fails.  Crucially a failed
   ``visible:`` binding falls back to the default — **true** — so controls that
   should be hidden appear instead.  The Stop button for a run that wasn't
   running showed up that way, and nothing in the log said why.

   ``win._firefly_ctx`` is also what the smoke fixture walks to stop
   controller-owned timers at teardown, so an object missing from it leaks
   timers as well as dying early.

2. An icon name with no SVG behind it renders as empty space.  ``crop`` and
   ``square`` did not exist; neither do ``trash-2`` or ``file-plus``, which
   three long-standing buttons still asked for.

Neither raises.  The shell tests share the one carefully-torn-down window from
``test_qml_smoke``: building the full shell ad hoc inside a full-suite run
deadlocks the Qt process.
"""
import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

pytest.importorskip("PySide6")

# Reused verbatim: it isolates HOME + every QSettings domain, silences the
# resource probes, and dismantles the window/controllers in the right order.
from test_qml_smoke import _app, qml_window  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QML_DIR = os.path.join(ROOT, "firefly", "ui", "qml")
ICON_DIR = os.path.join(QML_DIR, "assets", "icons")


def _qml_files():
    for dirpath, _dirnames, files in os.walk(QML_DIR):
        for f in sorted(files):
            if f.endswith(".qml"):
                yield os.path.join(dirpath, f)


# ── every icon name must have an SVG behind it ───────────────────────────────
def test_every_icon_name_in_qml_exists():
    """A missing icon is invisible, not an error — the control just renders with
    a blank where its glyph should be.  Static, so it needs no shell."""
    available = {os.path.splitext(f)[0] for f in os.listdir(ICON_DIR)
                 if f.endswith(".svg")}
    assert available, "the icon set is empty — has assets/icons moved?"

    missing = []
    # `icon: "name"` and `icon: cond ? "a" : "b"` both appear in this codebase
    pattern = re.compile(r'\bicon:\s*(.+)$', re.MULTILINE)
    literal = re.compile(r'"([a-z0-9][a-z0-9-]*)"')
    for path in _qml_files():
        src = open(path, encoding="utf-8").read()
        for line_no, line in enumerate(src.splitlines(), 1):
            m = pattern.search(line)
            if not m:
                continue
            for name in literal.findall(m.group(1)):
                if name and name not in available:
                    missing.append(
                        f"{os.path.relpath(path, ROOT)}:{line_no}: "
                        f'icon "{name}" has no SVG')
    assert not missing, "\n".join(missing)


# ── every context property must outlive build_main_window ────────────────────
def _registered_names():
    """The context-property names the app registers, read from the source so a
    newly added one is covered here automatically."""
    src = open(os.path.join(ROOT, "firefly", "ui", "app_qml.py"),
               encoding="utf-8").read()
    return set(re.findall(r'ctx\.setContextProperty\(\s*"([A-Za-z_]+)"', src))


def test_every_context_property_survives_the_builder(qml_window):
    """Registration is not enough: without a Python reference the object is
    collected the moment the builder returns and QML silently sees null."""
    _win, qw = qml_window
    ctx = qw.rootContext()
    names = _registered_names()
    assert {"Postproc", "Roi", "Vis"} <= names, sorted(names)

    dead = [n for n in sorted(names)
            if n != "appVersion"            # a plain string, not a QObject
            and ctx.contextProperty(n) is None]
    assert not dead, ("context properties collected before QML could use them "
                      f"(add them to win._firefly_ctx): {dead}")


def test_the_roi_section_is_wired_to_a_live_controller(qml_window):
    """The specific regression: Postproc.running must be readable, or the Stop
    button's `visible` binding fails open and it shows when nothing is running."""
    _win, qw = qml_window
    pp = qw.rootContext().contextProperty("Postproc")
    assert pp is not None, "Postproc was collected — QML sees null"
    assert pp.property("running") is False, (
        "nothing is running, so the Stop button must be hidden")
    vis = qw.rootContext().contextProperty("Vis")
    assert vis is not None and vis.property("openRunDir") == "", (
        "no run is open, so the Edit-ROI button must be hidden")


def test_the_keepalive_tuple_holds_every_registered_controller(qml_window):
    """Belt and braces, and it names the offender rather than just failing:
    _firefly_ctx is also what teardown walks to stop controller timers."""
    win, qw = qml_window
    ctx = qw.rootContext()
    kept = set(map(id, win._firefly_ctx))
    absent = []
    for name in sorted(_registered_names()):
        if name == "appVersion":
            continue
        obj = ctx.contextProperty(name)
        if obj is not None and id(obj) not in kept:
            absent.append(name)
    assert not absent, (
        f"registered but not held in win._firefly_ctx: {absent} — these survive "
        f"only by luck (an image provider happening to hold a bound method) and "
        f"their timers are never stopped at teardown")


# ── a binding that yields undefined silently shows the control ───────────────
def test_the_roi_editor_binds_without_undefined_assignments(qml_window, tmp_path):
    """`visible: obj && obj.msg && obj.msg.length > 0` evaluates to `undefined`
    when `obj.msg` is missing.  QML does not fail loudly: it drops the binding
    and the property keeps its DEFAULT, and for `visible` that default is TRUE —
    so the control appears when it was meant to be hidden.  In the ROI panel
    this rendered as an empty red error box above the detection preview.

    Qt reports it as "Unable to assign [undefined] to bool", so assert on Qt's
    own warning stream: that catches the whole class wherever it occurs, with
    none of the false positives a static scan produces (most `a && b` bindings
    here are QObject bool properties, which can never be undefined).

    NB the QML disk cache must be off — a cached compile of an earlier revision
    reports stale line numbers and masks the fix.
    """
    import numpy as np
    import tifffile
    from PySide6.QtCore import qInstallMessageHandler

    messages = []
    previous = qInstallMessageHandler(
        lambda mode, ctx, msg: messages.append(str(msg)))
    try:
        win, qw = qml_window
        roi = qw.rootContext().contextProperty("Roi")
        path = tmp_path / "preview.tif"
        frame = np.random.default_rng(3).normal(100, 1, (64, 64)).astype("float32")
        tifffile.imwrite(path, np.repeat(frame[None], 3, axis=0),
                         photometric="minisblack")
        roi.editFile(str(path))
        win.resize(1400, 950); win.show(); _app.processEvents()
        roi.detectEnabled = True
        _app.processEvents()
        win.hide()
    finally:
        qInstallMessageHandler(previous)

    offenders = [m for m in messages if "Unable to assign [undefined]" in m
                 or "Unable to assign [null] to bool" in m]
    assert not offenders, (
        "a binding evaluated to undefined/null and fell back to the property's "
        f"default (visible defaults to TRUE): {offenders[:4]}")


def _visible_texts(item, out=None):
    """Every effectively-visible `text` in an item subtree.

    `isVisible()` accounts for ancestors, so a label inside a hidden panel is
    correctly absent — which is the whole thing under test.
    """
    out = [] if out is None else out
    for child in item.childItems():
        try:
            value = child.property("text")
        except Exception:
            value = None
        if isinstance(value, str) and value and child.isVisible():
            out.append(value)
        _visible_texts(child, out)
    return out


def test_the_two_roi_panels_show_only_their_own_controls(qml_window, tmp_path):
    """Drawing a region and picking a detection threshold are separate screens
    of one viewer, reached by separate buttons (`editFile` / `editDetection`).

    They were one scrolling column, so each visit scrolled past the other job's
    controls.  Assert the split where it matters — what is actually on screen —
    because `Roi.panel` being right while both panels still render would look
    identical from Python.
    """
    import numpy as np
    import tifffile

    win, qw = qml_window
    roi = qw.rootContext().contextProperty("Roi")
    path = tmp_path / "panels.tif"
    frame = np.random.default_rng(11).normal(100, 1, (64, 64)).astype("float32")
    tifffile.imwrite(path, np.repeat(frame[None], 3, axis=0), photometric="minisblack")
    win.resize(1400, 950); win.show()

    try:
        roi.editFile(str(path))
        _app.processEvents()
        drawing = _visible_texts(qw.rootObject())

        roi.cancel(); _app.processEvents()
        roi.editDetection(str(path))
        _app.processEvents()
        threshold = _visible_texts(qw.rootObject())
    finally:
        roi.cancel()
        win.hide()
        _app.processEvents()

    assert "ROI MODE" in drawing and "Drawing tool" in drawing
    assert "DETECTION THRESHOLD" not in drawing, \
        "the ROI screen still carries the threshold controls"

    assert "DETECTION THRESHOLD" in threshold
    assert "ROI MODE" not in threshold and "Drawing tool" not in threshold, \
        "the threshold screen still carries the ROI controls"

    # each screen names what its primary button saves
    assert "Save ROI" in drawing and "Save threshold" not in drawing
    assert "Save threshold" in threshold and "Save ROI" not in threshold


def test_every_tab_renders_without_a_qml_script_error(qml_window):
    """A binding that throws does not break the page — it silently keeps its
    default and logs to a stream nobody reads.

    Found this way: `present` declared on a Badge and a Card but read from
    inside their `transform: Translate {}`.  A Translate is not an Item, so the
    bare name does not resolve from it; every visit threw a ReferenceError and
    the slide-in never ran.  Only a non-root object's properties have this
    problem — a component root's are in scope for everything in the component —
    which is why it looks fine until it isn't.

    Asserting Qt's own message stream catches the whole class on every page.
    """
    from PySide6.QtCore import qInstallMessageHandler

    win, qw = qml_window
    app_ctrl = qw.rootContext().contextProperty("App")
    importc = qw.rootContext().contextProperty("Import")
    messages = []
    previous = qInstallMessageHandler(
        lambda mode, ctx, msg: messages.append(str(msg)))
    try:
        win.resize(1400, 950); win.show()
        for batch in (False, True):
            importc.setBatchMode(batch)
            for tab in range(5):
                app_ctrl.enterMain(tab)
                for _ in range(6):
                    _app.processEvents()
        app_ctrl.goLanding()
        _app.processEvents()
    finally:
        qInstallMessageHandler(previous)
        win.hide()

    errors = [m for m in messages
              if any(k in m for k in ("ReferenceError", "TypeError",
                                      "is not defined", "is not a function"))]
    assert not errors, f"QML script errors while rendering the tabs: {errors[:4]}"


# ── QML must only reference things that exist ───────────────────────────────
# Neither of these fails loudly.  A missing controller member reads as
# `undefined` (and a `visible:` bound to it falls back to TRUE); a missing
# function on an id throws a TypeError that aborts the rest of its handler.
# `guide.rescore()` did the latter from rc.8 to rc.12: the threshold histogram
# never followed the slider and the detection overlay never re-ran.
_CONTEXT = ["Theme", "App", "Settings", "Import", "Process", "Analysis", "Vis", "Roi",
            "Postproc", "Embed", "Sidebar", "Preset", "Batch", "Hyperfly", "Updates", "Cuda"]


def _meta_members(obj):
    from PySide6.QtCore import QMetaMethod
    mo = obj.metaObject()
    props = {mo.property(i).name() for i in range(mo.propertyCount())}
    writable = {mo.property(i).name() for i in range(mo.propertyCount())
                if mo.property(i).isWritable()}
    methods, signals = set(), set()
    for i in range(mo.methodCount()):
        m = mo.method(i)
        name = bytes(m.name()).decode()
        methods.add(name)
        if m.methodType() == QMetaMethod.MethodType.Signal:
            signals.add(name)
    return props, writable, methods, signals


def test_every_controller_member_qml_uses_exists(qml_window):
    import pathlib
    from _qml_source import block_end, strip_qml

    _win, qw = qml_window
    api = {}
    for name in _CONTEXT:
        obj = qw.rootContext().contextProperty(name)
        assert obj is not None, f"{name} is null in QML"
        api[name] = _meta_members(obj)

    use = re.compile(r'\b(' + "|".join(_CONTEXT) + r')\.([A-Za-z_]\w*)')
    assign = re.compile(r'\b(' + "|".join(_CONTEXT) + r')\.([A-Za-z_]\w*)\s*=(?!=)')
    missing, readonly, handlers = [], [], []
    for path in _qml_files():
        clean = strip_qml(pathlib.Path(path).read_text(encoding="utf-8"))
        rel = os.path.relpath(path, ROOT)
        for ln, line in enumerate(clean.split("\n"), 1):
            for ctx, member in use.findall(line):
                props, _w, methods, _s = api[ctx]
                if member not in props and member not in methods:
                    missing.append(f"{rel}:{ln} {ctx}.{member}")
            for ctx, member in assign.findall(line):
                props, writable, _m, _s = api[ctx]
                if member in props and member not in writable:
                    readonly.append(f"{rel}:{ln} {ctx}.{member} = …")
        for m in re.finditer(r'Connections\s*\{', clean):
            block = clean[m.end():block_end(clean, m.end() - 1)]
            t = re.search(r'target:\s*(\w+)', block)
            if not t or t.group(1) not in api:
                continue
            for h in re.findall(r'function\s+on([A-Z]\w*)\s*\(', block):
                if h[0].lower() + h[1:] not in api[t.group(1)][3]:
                    handlers.append(f"{rel} Connections→{t.group(1)}: on{h}")

    assert not missing, f"QML uses controller members that do not exist: {missing}"
    assert not readonly, f"QML assigns read-only controller properties: {readonly}"
    assert not handlers, f"Connections handlers for signals that do not exist: {handlers}"


def test_every_member_qml_reads_off_an_id_exists(qml_window):
    """`someId.member` must exist on that id's object: what its block declares,
    or what its type provides — read from a real instance in the app's engine,
    so built-ins (Timer, Canvas…) and our own components are both covered."""
    import pathlib
    from PySide6.QtCore import QUrl
    from PySide6.QtQml import QQmlComponent
    from _qml_source import block_end, strip_qml

    _win, qw = qml_window
    engine = qw.engine()
    header = ('import QtQuick\nimport QtQuick.Layouts\nimport QtQuick.Controls as QQC\n'
              'import QtQuick.Shapes\nimport "components"\nimport "."\n')
    cache = {}

    def type_members(tname):
        if tname not in cache:
            comp = QQmlComponent(engine)
            comp.setData(f"{header}{tname} {{}}\n".encode(),
                         QUrl.fromLocalFile(os.path.join(QML_DIR, "_probe.qml")))
            obj = comp.create(qw.rootContext())
            if obj is None:
                cache[tname] = None
            else:
                props, _w, methods, _s = _meta_members(obj)
                cache[tname] = props | methods
                obj.deleteLater()
        return cache[tname]

    findings = []
    for path in _qml_files():
        clean = strip_qml(pathlib.Path(path).read_text(encoding="utf-8"))
        ids = {}
        for m in re.finditer(r'\b((?:QQC\.)?[A-Z]\w*)\s*\{', clean):
            body = clean[m.end():block_end(clean, m.end() - 1)]
            top, depth = [], 0                     # this block's direct members only
            for ch in body:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                elif depth == 0:
                    top.append(ch)
            top = "".join(top)
            idm = re.search(r'(?:^|[\s;])id\s*:\s*(\w+)', top)
            if not idm:
                continue
            declared = set(re.findall(
                r'\bproperty\s+(?:alias\s+|var\s+|[\w.<>]+\s+)?(\w+)\s*[:\n;]', top))
            declared |= set(re.findall(r'\bfunction\s+(\w+)\s*\(', top))
            declared |= set(re.findall(r'\bsignal\s+(\w+)', top))
            # an id may recur in separate delegate scopes; any of them will do
            ids.setdefault(idm.group(1), []).append((m.group(1), declared))
        rel = os.path.relpath(path, ROOT)
        for ln, line in enumerate(clean.split("\n"), 1):
            for idn, member in re.findall(r'\b([a-z_]\w*)\.([A-Za-z_]\w*)', line):
                if idn not in ids:
                    continue
                ok = unknown = False
                for tname, declared in ids[idn]:
                    if member in declared:
                        ok = True
                        break
                    members = type_members(tname)
                    if members is None:            # inline component: not probeable
                        unknown = True
                    elif member in members:
                        ok = True
                        break
                if not ok and not unknown:
                    findings.append(f"{rel}:{ln} {idn}.{member}")

    assert not findings, f"QML reads members that do not exist on the id: {findings}"
