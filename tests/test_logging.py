"""Tests for the persistent-logging infrastructure (crash_reporter)."""
import logging

import crash_reporter as cr


def _reset_handlers(saved):
    root = logging.getLogger()
    for h in root.handlers[:]:
        if h not in saved:
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass


def test_setup_logging_writes_to_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "crash_report_dir",
                        lambda: str(tmp_path / "crash_reports"))
    monkeypatch.setattr(cr, "_logging_configured", False)
    saved = logging.getLogger().handlers[:]
    try:
        path = cr.setup_logging(console=False, filename="t.log")
        assert path.endswith("t.log")
        cr.get_logger().error("hello-firefly-log")
        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:
                pass
        with open(path, encoding="utf-8") as fh:
            assert "hello-firefly-log" in fh.read()
    finally:
        _reset_handlers(saved)


def test_log_exception_records_traceback(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "crash_report_dir", lambda: str(tmp_path / "cr"))
    monkeypatch.setattr(cr, "_logging_configured", False)
    saved = logging.getLogger().handlers[:]
    try:
        path = cr.setup_logging(console=False, filename="exc.log")
        try:
            raise ValueError("boom-xyz")
        except ValueError:
            cr.log_exception("caught it")
        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:
                pass
        text = open(path, encoding="utf-8").read()
        assert "caught it" in text
        assert "boom-xyz" in text
        assert "ValueError" in text
    finally:
        _reset_handlers(saved)


def test_log_dir_is_sibling_of_crash_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "crash_report_dir",
                        lambda: str(tmp_path / "FIREFLY" / "crash_reports"))
    d = cr.log_dir()
    assert d.endswith("logs")
    assert "FIREFLY" in d
