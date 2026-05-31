"""Unit tests for the CUDA sidecar installer's pure logic (no network)."""
import os

import pytest

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


# ── configurable install location ─────────────────────────────────────────────
def _fake_install(extracted_torch_dir):
    os.makedirs(extracted_torch_dir, exist_ok=True)
    open(os.path.join(extracted_torch_dir, "__init__.py"), "w").close()


def test_sidecar_base_default_and_pointer_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cu, "_firefly_app_dir", lambda: str(tmp_path))
    # Default base lives under the fixed anchor.
    assert cu.sidecar_base() == os.path.join(str(tmp_path), "torch-cuda")
    # sidecar_dir is base/<cpXX>.
    assert os.path.basename(cu.sidecar_dir()) == cu._current_py_tag()
    # Pointer round-trips and overrides the default.
    custom = os.path.join(str(tmp_path), "elsewhere", "torch-cuda")
    cu.set_sidecar_base(custom)
    assert cu.sidecar_base() == custom
    assert cu.sidecar_dir().startswith(custom)
    # Clearing reverts to default.
    cu.set_sidecar_base(None)
    assert cu.sidecar_base() == os.path.join(str(tmp_path), "torch-cuda")


def test_move_install_relocates_tree_and_updates_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr(cu, "_firefly_app_dir", lambda: str(tmp_path / "anchor"))
    _fake_install(os.path.join(cu.sidecar_dir(), "extracted", "torch"))
    assert cu.is_installed()

    dest_parent = str(tmp_path / "dest")
    new_base = cu.move_install(dest_parent)

    assert new_base == os.path.join(dest_parent, "torch-cuda")
    assert cu.sidecar_base() == new_base                    # pointer updated
    assert os.path.isfile(os.path.join(            # tree physically moved
        new_base, cu._current_py_tag(), "extracted", "torch", "__init__.py"))
    assert not os.path.exists(                              # old base gone
        os.path.join(str(tmp_path / "anchor"), "torch-cuda"))
    assert cu.is_installed()                                # still installed


def test_move_install_rejects_nonempty_target(tmp_path, monkeypatch):
    monkeypatch.setattr(cu, "_firefly_app_dir", lambda: str(tmp_path / "anchor"))
    _fake_install(os.path.join(cu.sidecar_dir(), "extracted", "torch"))
    busy = os.path.join(str(tmp_path / "dest"), "torch-cuda", "junk")
    os.makedirs(busy, exist_ok=True)
    open(os.path.join(busy, "f.txt"), "w").close()
    with pytest.raises(RuntimeError):
        cu.move_install(str(tmp_path / "dest"))


# ── wheel discovery (pure, no network) ────────────────────────────────────────
_SAMPLE_INDEX = """
<a href="cu130/torch-2.12.0%2Bcu130-cp313-cp313-win_amd64.whl#sha256=a">w</a>
<a href="cu130/torch-2.10.0%2Bcu130-cp313-cp313-win_amd64.whl">w</a>
<a href="cu130/torch-2.5.1%2Bcu130-cp313-cp313-win_amd64.whl">out of range</a>
<a href="cu130/torch-2.12.0%2Bcu130-cp312-cp312-win_amd64.whl">wrong python</a>
<a href="cu130/torch-2.12.0%2Bcu130-cp313-cp313-linux_x86_64.whl">wrong os</a>
"""


def test_extract_wheel_versions_filters_and_sorts():
    v = cu._extract_wheel_versions(_SAMPLE_INDEX, "cu130", "cp313")
    # 2.5.1 is below the >=2.6 floor; cp312 + linux entries are excluded.
    assert v == ["2.12.0", "2.10.0"]


def test_extract_wheel_versions_wrong_python_tag_empty():
    assert cu._extract_wheel_versions(_SAMPLE_INDEX, "cu130", "cp311") == []


def test_select_version_prefers_exact_then_newest():
    avail = ["2.12.0", "2.10.0", "2.8.0"]
    assert cu._select_version(avail, "2.10.0") == "2.10.0"   # exact match
    assert cu._select_version(avail, "2.99.0") == "2.12.0"   # absent -> newest
    assert cu._select_version(avail, None) == "2.12.0"
    assert cu._select_version([], "2.10.0") is None


def test_version_range_bounds():
    assert cu._ver_in_range(cu._parse_ver("2.6.0")) is True
    assert cu._ver_in_range(cu._parse_ver("2.12.0")) is True
    assert cu._ver_in_range(cu._parse_ver("2.5.1")) is False   # below floor
    assert cu._ver_in_range(cu._parse_ver("3.0.0")) is False   # at ceiling
    assert cu._ver_in_range(cu._parse_ver("nope")) is False


def test_cuda_tags_are_newest_first():
    # cu130 must be probed before the legacy cu124/cu121/cu118 so a modern
    # GPU gets the newest toolkit, and torch versions that only ship on the
    # newest channel (e.g. 2.12.0 on cu130) are found at all.
    tags = cu._CUDA_TAGS_NEWEST_FIRST
    nums = [int(t[2:]) for t in tags]
    assert nums == sorted(nums, reverse=True)
    assert tags[0] == "cu130"
