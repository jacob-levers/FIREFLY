"""Headless self-test harness (diagnostics only).

Runs ONE analysis through the real ``firefly_worker.run_analysis`` subprocess —
exactly as the GUI spawns it (spawn context, real Queue/Event, GUI-default
params) — but drains the message queue itself, writing every line (especially
the ``MsgKind.ERROR`` traceback the GUI log may swallow) to a durable file.

Triggered by ``FIREFLY_SELFTEST=<input-file>``; never runs on a normal launch.
Optional env knobs: ``FIREFLY_SELFTEST_OUT`` (output dir),
``FIREFLY_SELFTEST_WORKERS`` (override worker count for fast iteration),
``FIREFLY_SELFTEST_BACKEND`` (override detection backend).

Crucially, this exercises the FROZEN code path: when run from the packaged
.exe, the worker (and its nested localisation pools) are spawned by re-launching
the bundle — the precise scenario that fails only in the Windows frozen build.
"""
from __future__ import annotations

import os
import sys
import time


def run_selftest(input_file: str) -> int:
    import multiprocessing as mp
    from firefly import crash_reporter
    from firefly import firefly_worker
    from firefly.analysis.fa_enums import MsgKind
    from firefly.ui.controllers.params.params_builder import build_params

    log_path = os.path.join(crash_reporter.log_dir(), "selftest.log")
    out_dir = os.environ.get("FIREFLY_SELFTEST_OUT") or os.path.join(
        os.path.dirname(crash_reporter.log_dir()), "selftest_out")
    os.makedirs(out_dir, exist_ok=True)
    logf = open(log_path, "w", encoding="utf-8", buffering=1)

    def w(msg):
        try:
            logf.write(str(msg) + "\n")
        except Exception:
            pass
        try:                       # console is /dev/null in the frozen app; harmless
            print(msg, flush=True)
        except Exception:
            pass

    w(f"[selftest] python={sys.version.split()[0]} frozen={getattr(sys, 'frozen', False)} "
      f"meipass={getattr(sys, '_MEIPASS', None)!r}")
    w(f"[selftest] input={input_file!r} out={out_dir!r}")

    class _FS:                     # fake SettingsController → returns every default
        def get_str(self, k, d):   return d
        def get_float(self, k, d): return d
        def get_bool(self, k, d):  return d

    class _FI:                     # fake ImportController
        filePath = input_file; outDir = out_dir
        pixelSize = 0.106; frameInterval = 0.02
        overridePx = False; overrideFi = False; isCsv = False
        csvPreset = "auto"; bgImagePath = ""

    try:
        params = build_params(_FS(), _FI(), input_file, out_dir)
    except Exception:
        import traceback
        w("[selftest] build_params FAILED:\n" + traceback.format_exc())
        logf.close()
        return 2

    wk = os.environ.get("FIREFLY_SELFTEST_WORKERS")
    if wk:
        try:    params["workers"] = int(wk)
        except ValueError: pass
    bk = os.environ.get("FIREFLY_SELFTEST_BACKEND")
    if bk:
        params["backend"] = bk
    w(f"[selftest] backend={params.get('backend')!r} workers={params.get('workers')!r} "
      f"linker={params.get('linker')!r} auto_minmass={params.get('auto_minmass')!r}")

    q = mp.Queue(maxsize=2000)
    ev = mp.Event()
    p = mp.Process(target=firefly_worker.run_analysis, args=(params, q, ev),
                   name="FIREFLY-SelfTestWorker", daemon=False)
    t0 = time.time()
    p.start()
    saw_error = saw_done = False
    while True:
        try:
            kind, payload = q.get(timeout=1.0)
        except Exception:
            if not p.is_alive():
                break
            continue
        if kind == MsgKind.LOG:
            w("LOG: " + str(payload))
        elif kind == MsgKind.ERROR:
            saw_error = True
            w("\n==== WORKER ERROR (MsgKind.ERROR) ====\n" + str(payload)
              + "\n==== END WORKER ERROR ====")
        elif kind == MsgKind.DONE:
            saw_done = True
            w("==== DONE ====")
        elif kind == MsgKind.STOPPED:
            w("==== STOPPED ====")
        # PROGRESS / MASS_CHUNK / PREVIEW_FRAME intentionally dropped.
    p.join(timeout=15)
    w(f"[selftest] EXITCODE={p.exitcode} elapsed={time.time() - t0:.1f}s "
      f"saw_error={saw_error} saw_done={saw_done}")
    logf.close()
    return 0 if (p.exitcode == 0 and saw_done and not saw_error) else 1
