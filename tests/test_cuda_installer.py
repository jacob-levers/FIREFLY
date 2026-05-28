"""Unit tests for the CUDA sidecar installer's pure logic (no network)."""
import os

from firefly import cuda_installer as cu


def test_cuda_wheel_url_format():
    url = cu.cuda_wheel_url("2.5.1", cuda_tag="cu124", python_tag="cp312")
    assert url == (
        "https://download.pytorch.org/whl/cu124/"
        "torch-2.5.1%2Bcu124-cp312-cp312-win_amd64.whl")


def test_cuda_wheel_url_respects_tag_and_python():
    url = cu.cuda_wheel_url("2.5.1", cuda_tag="cu121", python_tag="cp311")
    assert "/cu121/" in url
    assert "cp311-cp311-win_amd64.whl" in url
    assert "%2Bcu121" in url  # '+' is URL-encoded


def test_sidecar_paths_are_strings():
    assert isinstance(cu.sidecar_dir(), str)
    assert cu.sidecar_extracted_dir().endswith("extracted")


def test_sidecar_dir_is_version_namespaced():
    """The sidecar root must carry the interpreter ABI tag so a CUDA wheel
    installed under one Python version can't be picked up by another (the
    bug that broke `import torch` after the 3.13 bump)."""
    tag = cu._current_py_tag()
    assert tag == f"cp{__import__('sys').version_info.major}" \
                  f"{__import__('sys').version_info.minor}"
    assert os.path.basename(cu.sidecar_dir()) == tag


def _make_fake_torch(extracted, ext_filename):
    """Create extracted/torch/{__init__.py, <ext_filename>}."""
    tdir = os.path.join(extracted, "torch")
    os.makedirs(tdir, exist_ok=True)
    open(os.path.join(tdir, "__init__.py"), "w").close()
    if ext_filename:
        open(os.path.join(tdir, ext_filename), "w").close()
    return extracted


def test_abi_ok_accepts_matching_interpreter(tmp_path):
    tag = cu._current_py_tag()
    extracted = _make_fake_torch(str(tmp_path), f"_C.{tag}-win_amd64.pyd")
    assert cu._sidecar_abi_ok(extracted) is True


def test_abi_ok_rejects_mismatched_interpreter(tmp_path):
    # A cp312 binary must be rejected when we're not running cp312.
    other = "cp312" if cu._current_py_tag() != "cp312" else "cp311"
    extracted = _make_fake_torch(str(tmp_path), f"_C.{other}-win_amd64.pyd")
    assert cu._sidecar_abi_ok(extracted) is False


def test_is_installed_false_for_mismatched_abi(tmp_path, monkeypatch):
    other = "cp312" if cu._current_py_tag() != "cp312" else "cp311"
    sd = tmp_path / "torch-cuda" / cu._current_py_tag()
    extracted = sd / "extracted"
    _make_fake_torch(str(extracted), f"_C.{other}-win_amd64.pyd")
    monkeypatch.setattr(cu, "sidecar_dir", lambda: str(sd))
    # torch/__init__.py exists, but the ABI is wrong -> not installed.
    assert cu.is_installed() is False


def test_is_windows_returns_bool():
    assert isinstance(cu.is_windows(), bool)
