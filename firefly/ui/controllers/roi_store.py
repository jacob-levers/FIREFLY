"""RoiStore — per-file manual-polygon ROI store (Phase 6c).

Mirrors the Widgets ``self._roi_polygons`` dict: maps an input file's absolute
path to its drawn polygon(s) (each a list of ``(y, x)`` vertices).  Shared
between RoiController (writes when the user draws) and params_builder (reads into
the ``roi_polygon`` key — sent regardless of ROI mode, matching
``_build_params_for_file``).  Plain dict-like; no Qt.

Given a ``path``, both stores here also save to a JSON file there (the app
keeps them in FIREFLY's data folder), so ROIs survive closing the app.  Each
entry records the recording's size and modification time: an ROI is pixel
coordinates on ONE recording, so a different file later found at the same path
does not inherit it.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

_FORMAT_VERSION = 1


def _signature(path):
    """``[size, mtime]`` of a recording, or None when it cannot be checked (a
    directory — run folders change as they are written — or a file that is
    not there, e.g. on an unplugged drive)."""
    try:
        if not os.path.isfile(path):
            return None
        st = os.stat(path)
        return [int(st.st_size), int(st.st_mtime)]
    except OSError:
        return None


class _Durable:
    """In-memory ``abspath → value`` map, optionally mirrored to a JSON file."""

    def __init__(self, path=None):
        self._by_file: dict = {}
        self._sig: dict = {}
        self._path = str(path) if path else None
        if self._path:
            self._load()

    @staticmethod
    def _key(path):
        return os.path.abspath(path) if path else ""

    # values as stored in JSON ↔ as handed out
    def _encode(self, value):
        return value

    def _decode(self, value):
        return value

    def _load(self):
        try:
            with open(self._path, encoding="utf-8") as fh:
                data = json.load(fh)
            files = data["files"]
            if not isinstance(files, dict):
                raise ValueError("no files table")
        except FileNotFoundError:
            return
        except Exception:
            # Never overwrite what cannot be read: set it aside and start empty.
            try:
                os.replace(self._path, f"{self._path}.unreadable-{time.strftime('%Y%m%d-%H%M%S')}")
            except OSError:
                pass
            return
        for k, rec in files.items():
            try:
                self._by_file[k] = self._decode(rec["value"])
                self._sig[k] = rec.get("signature")
            except Exception:
                continue                     # one bad record does not cost the rest

    def _save(self):
        if not self._path:
            return
        tmp = None
        try:
            directory = os.path.dirname(self._path) or "."
            os.makedirs(directory, exist_ok=True)
            data = {"version": _FORMAT_VERSION,
                    "files": {k: {"value": self._encode(v), "signature": self._sig.get(k)}
                              for k, v in self._by_file.items()}}
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=directory,
                    prefix=".roi-", suffix=".tmp", delete=False) as fh:
                tmp = fh.name
                json.dump(data, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._path)
        except Exception:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def _current(self, path):
        """The entry for ``path`` — unless it was made for another recording."""
        k = self._key(path)
        if k not in self._by_file:
            return None, k
        saved, now = self._sig.get(k), _signature(k)
        if saved is not None and now is not None and list(saved) != list(now):
            return None, k
        return self._by_file[k], k

    def _put(self, path, value):
        k = self._key(path)
        self._by_file[k] = value
        self._sig[k] = _signature(k)
        self._save()

    def _drop(self, path):
        if self._by_file.pop(self._key(path), None) is not None:
            self._sig.pop(self._key(path), None)
            self._save()


class RoiStore(_Durable):
    def _encode(self, polygons):
        return [[[y, x] for y, x in poly] for poly in polygons]

    def _decode(self, polygons):
        return [[(float(y), float(x)) for y, x in poly] for poly in polygons]

    def get(self, path):
        """Polygons for a file as ``[[(y, x), …], …]`` or None."""
        return self._current(path)[0]

    def set(self, path, polygons):
        if polygons:
            self._put(path, [[(float(y), float(x)) for y, x in poly]
                             for poly in polygons])
        else:
            self._drop(path)

    def has(self, path):
        return bool(self._current(path)[0])

    def clear(self, path):
        self._drop(path)


class RoiOverrideStore(_Durable):
    """Per-file analysis override (abspath → spec dict), read by
    ``params_builder`` at run time.  Saved with ``RoiStore`` (see the module note).

    Carries the ``analysis/roi_*`` values in their settings-label form
    (``roi_mode`` / ``roi_auto_method`` / ``roi_threshold`` / ``roi_mask_mode`` /
    ``roi_bg_sigma``, plus ``roi_split_replicates`` / ``roi_labels``) and,
    optionally, a per-file DETECTION threshold: ``minmass`` and ``auto_minmass``.

    The left sidebar holds the default applied to every file; a file with an
    entry here overrides that default for THAT file only (set from the Preview &
    ROI viewer).  A per-file minmass is deliberately opt-in per file rather than
    a mode: in a batch it lets one recording be thresholded differently from the
    rest, which is powerful and easy to misuse — mass is file-relative (frames
    are min-max normalised), so the same number is not the same brightness in
    two files, but tuning each file by eye until the counts match is how a
    detection difference gets manufactured.  Every run records the value it
    used, and Compare warns when one group mixes thresholds.
    """

    def get(self, path):
        return self._current(path)[0]

    def set(self, path, spec):
        if spec:
            self._put(path, dict(spec))
        else:
            self._drop(path)

    def has(self, path):
        return self._current(path)[0] is not None

    def clear(self, path):
        self._drop(path)
