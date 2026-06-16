"""Deterministic execution mode (FIREFLY_DETERMINISTIC) and its run-manifest
record.  All of FIREFLY's RNG is already constant-seeded; this mode adds the
torch-side determinism switch and discloses the live state in the manifest."""
import json
import pytest


def test_apply_determinism_noop_when_unset(monkeypatch):
    import firefly.firefly_worker as fw
    monkeypatch.delenv("FIREFLY_DETERMINISTIC", raising=False)
    assert fw._apply_determinism() == {"requested": False}


def test_flag_enables_deterministic_algorithms(monkeypatch):
    torch = pytest.importorskip("torch")
    import firefly.firefly_worker as fw
    monkeypatch.setenv("FIREFLY_DETERMINISTIC", "1")
    monkeypatch.setattr(fw, "_DET_APPLIED", False, raising=False)
    try:
        st = fw._apply_determinism()
        assert st.get("applied") is True
        assert torch.are_deterministic_algorithms_enabled() is True
    finally:
        # don't leak global determinism state into other tests
        torch.use_deterministic_algorithms(False)


def test_manifest_records_determinism(tmp_path, monkeypatch):
    import firefly.firefly_worker as fw
    f = tmp_path / "stack.tif"
    f.write_bytes(b"not a real tif")
    monkeypatch.setenv("FIREFLY_DETERMINISTIC", "1")
    path = fw._write_run_manifest(out_dir=str(tmp_path), stem="stack",
                                  fpath=str(f), params={"diameter": 7})
    m = json.load(open(path, encoding="utf-8"))
    assert m["schema_version"] == 2
    assert "determinism" in m
    det = m["determinism"]
    assert det["requested"] is True
    assert "numpy_version" in det and det["numpy_version"]


def test_manifest_determinism_unset(tmp_path, monkeypatch):
    import firefly.firefly_worker as fw
    f = tmp_path / "stack.tif"
    f.write_bytes(b"x")
    monkeypatch.delenv("FIREFLY_DETERMINISTIC", raising=False)
    path = fw._write_run_manifest(out_dir=str(tmp_path), stem="stack",
                                  fpath=str(f), params={})
    m = json.load(open(path, encoding="utf-8"))
    assert m["determinism"]["requested"] is False
