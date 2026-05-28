"""Unit tests for the CUDA sidecar installer's pure logic (no network)."""
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


def test_is_windows_returns_bool():
    assert isinstance(cu.is_windows(), bool)
