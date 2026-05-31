"""MainWindow BatchMixin methods, split out of app_qt.py (#7)."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from firefly import sptpalm_analysis
from firefly import crash_reporter
from firefly.ui.ui_theme import _THEME
from firefly.ui.ui_constants import (TAB_IMPORT, TAB_ANALYSIS, TAB_COMPARE,
                          TAB_VISUALISE, TAB_REPROCESS)


class BatchMixin:
    def _batch_rescan(self, folder: str):
        """Populate the tree with one parent per file SERIES + one child
        per sister file inside the series.

        Each series's parent toggles all of its file children at once;
        the children can be individually deselected to exclude specific
        sister files from the loader concat.  When the run starts, the
        worker receives the per-series checked-file list and overrides
        the auto-discovery in `load_tif` / `load_czi` accordingly.
        """
        # Disconnect itemChanged so populating doesn't fire a cascade
        try:
            self.tree_batch_files.itemChanged.disconnect()
        except (TypeError, RuntimeError):
            pass
        # Re-entrancy guard for parent ↔ child propagation
        self._tree_propagation_guard = False

        self.tree_batch_files.blockSignals(True)
        self.tree_batch_files.clear()
        self._batch_series_map: dict[str, list[tuple[str, str]]] = {}

        if not os.path.isdir(folder):
            self.tree_batch_files.blockSignals(False)
            self._batch_update_summary()
            return
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            self.tree_batch_files.blockSignals(False)
            self._batch_update_summary()
            return

        # Phase 1 — collect candidate files from the picked folder AND
        # one level of subfolders (lab layouts like 1-AMA/Cell1/Loc.txt
        # are typical).  For files inside a subfolder we keep the
        # subfolder name in the display label so the user can see
        # which cell each row came from, and we prefix the series key
        # with the subfolder so e.g. `Cell1/Loc.txt` and `Cell2/Loc.txt`
        # don't collapse into a single ambiguous "Loc" series.
        # `candidates` holds (display_name, full_path, subfolder_or_None).
        # `self._batch_corrupt_paths` collects full-paths that fail the
        # cheap integrity probe so the tree-builder can flag + auto-skip
        # them.
        candidates: list[tuple[str, str, str | None]] = []
        self._batch_corrupt_paths: set[str] = set()
        for name in names:
            if name.startswith("."):
                continue
            full = os.path.join(folder, name)
            if os.path.isfile(full):
                if self._looks_like_input_file(name):
                    if self._file_looks_corrupt(full):
                        self._batch_corrupt_paths.add(full)
                    candidates.append((name, full, None))
            elif os.path.isdir(full):
                # Skip the batch_results / drift_correction output sub-
                # dirs we ourselves write, plus hidden folders.
                if name.lower() in ("batch_results", "compare_results"):
                    continue
                try:
                    child_names = sorted(os.listdir(full))
                except OSError:
                    continue
                for cname in child_names:
                    # Skip hidden / macOS AppleDouble (`._*`) and `.DS_Store`
                    # stubs — same guard the top-level loop applies.  Without
                    # it, Mac-exported folders surface 4 KB `._<name>.czi`
                    # ghosts that then get flagged corrupt and clutter the list.
                    if cname.startswith("."):
                        continue
                    cfull = os.path.join(full, cname)
                    if (os.path.isfile(cfull)
                        and self._looks_like_input_file(cname)):
                        if self._file_looks_corrupt(cfull):
                            self._batch_corrupt_paths.add(cfull)
                        display = f"{name}/{cname}"
                        candidates.append((display, cfull, name))

        # Phase 1c — prefer the raw acquisition.  If a folder contains a `.czi`,
        # drop sibling `.tif`/`.csv` files from that SAME folder: they're almost
        # always derived from the CZI (e.g. MotionCorrected.tif, Processed.tif,
        # an ImageJ Results.csv), not separate acquisitions, and queuing them
        # would re-analyse the same recording 3-4×.  Folders with no `.czi`
        # (pure TIFF or external-CSV datasets) are left untouched.
        from collections import defaultdict as _dd
        _by_folder: "dict[str|None, list]" = _dd(list)
        for _c in candidates:
            _by_folder[_c[2]].append(_c)
        _filtered = []
        for _grp in _by_folder.values():
            if any(c[1].lower().endswith(".czi") for c in _grp):
                _filtered.extend(c for c in _grp if c[1].lower().endswith(".czi"))
            else:
                _filtered.extend(_grp)
        candidates = _filtered

        # Phase 2 — group by series key.  Files in a subfolder get a
        # `<subfolder>__<base_key>` key so same-named files in different
        # cells stay separate; top-level files keep the bare base key.
        #
        # When the base key is a generic palmTRACER label (e.g.
        # `locPALMTracer`), the subfolder name IS the experiment
        # identifier on its own — repeating both produces hideous
        # `20260122_..._Post.PT__locPALMTracer/` output folders.  In
        # that case the subfolder alone is enough.
        GENERIC_PT_NAMES = {"locpalmtracer", "trcpalmtracer"}
        import re as _re_sub
        for display, full, sub in candidates:
            if sub is None:
                key = self._series_key(display)
            else:
                base = self._series_key(os.path.basename(display))
                # Sanitise the subfolder name for the output-folder path
                # (worker uses the series key verbatim as the stem).
                # Strip palmTRACER's `.PT` analysis-dir suffix so the
                # output folder doesn't carry the analysis-software
                # marker.
                sub_clean = _re_sub.sub(r"\.PT$", "", sub,
                                          flags=_re_sub.IGNORECASE)
                safe_sub = _re_sub.sub(r"[^A-Za-z0-9_.-]+", "_",
                                         sub_clean).strip("_") or "sub"
                if base.lower() in GENERIC_PT_NAMES:
                    # Subfolder alone is enough (the filename carries
                    # no extra identity over the directory it's in).
                    key = safe_sub
                else:
                    key = f"{safe_sub}__{base}"
            self._batch_series_map.setdefault(key, []).append(
                (display, full))

        # Phase 1b — drop ROI/channel sister files (palmTRACER's
        # `<base>_green.tif`, `<base>_red.tif`, etc.).  These have
        # their own series key of the form `<base>_<word>` — drop
        # them only when an unsuffixed `<base>` series also exists,
        # so a legitimate standalone `MyExperiment_green.tif` with
        # no companion stays put.
        import re as _re
        existing_keys = set(self._batch_series_map.keys())
        roi_keys_to_drop = []
        for key in list(self._batch_series_map.keys()):
            m = _re.match(r"^(.+)_[^_]+$", key)
            if m and m.group(1) in existing_keys:
                roi_keys_to_drop.append(key)
        for k in roi_keys_to_drop:
            self._batch_series_map.pop(k, None)

        # Natural-sort key for sibling files within a series.  Must
        # match `sptpalm_analysis._tif_series_nat_key` exactly — the
        # tree's display order is the same order the loader will
        # concatenate frames in, so lab members can verify the run
        # at a glance.  Bare `<root>.tif` first (key=-1), then
        # `-fileNNN` / `(N)` in numeric order.
        def _nat_key(name_full_pair):
            name = name_full_pair[0]
            ext  = os.path.splitext(name)[1]
            m_pt = _re.search(r"-file(\d+)" + _re.escape(ext) + r"$",
                              name, _re.IGNORECASE)
            if m_pt: return int(m_pt.group(1))
            m_ij = _re.search(r"\((\d+)\)" + _re.escape(ext) + r"$",
                              name, _re.IGNORECASE)
            if m_ij: return int(m_ij.group(1))
            return -1

        # Phase 2 — build the tree.
        for key in sorted(self._batch_series_map.keys()):
            sisters = sorted(self._batch_series_map[key], key=_nat_key)
            primary_name, primary_full = sisters[0]
            for nm, pth in sisters:
                if os.path.splitext(nm)[0] == key:
                    primary_name, primary_full = nm, pth
                    break
            n = len(sisters)
            corrupt = getattr(self, "_batch_corrupt_paths", set())
            # A series is wholly unusable when every sister file is corrupt.
            n_corrupt = sum(1 for _nm, _pth in sisters if _pth in corrupt)
            series_dead = (n_corrupt == n)
            parent_label = (primary_name if n == 1
                            else f"{primary_name}   ×  {n} files")
            if series_dead:
                parent_label = f"⚠ {parent_label}  — corrupt, skipped"
            parent = QtWidgets.QTreeWidgetItem([parent_label])
            parent.setFlags(parent.flags()
                            | Qt.ItemFlag.ItemIsUserCheckable
                            | Qt.ItemFlag.ItemIsAutoTristate)
            parent.setCheckState(0, Qt.CheckState.Unchecked
                                 if series_dead else Qt.CheckState.Checked)
            parent.setData(0, self._ROLE_PATH, primary_full)
            parent.setData(0, self._ROLE_KIND, "series")
            parent.setData(0, self._ROLE_SERIES_KEY, key)
            parent.setData(0, self._ROLE_FILE_COUNT, n)
            # Highlight the parent slightly to distinguish from children
            f = parent.font(0); f.setBold(True); parent.setFont(0, f)
            # Add one child per sister file (in display order)
            for nm, pth in sisters:
                is_corrupt = pth in corrupt
                label = f"⚠ {nm}  — corrupt (all-null / unreadable)" \
                    if is_corrupt else nm
                child = QtWidgets.QTreeWidgetItem([label])
                child.setFlags(child.flags()
                               | Qt.ItemFlag.ItemIsUserCheckable)
                # Corrupt files start UNCHECKED so they can't be run by
                # accident; the user can still tick them manually if they
                # really want the loader to try (and fail loudly).
                child.setCheckState(0, Qt.CheckState.Unchecked
                                    if is_corrupt else Qt.CheckState.Checked)
                child.setData(0, self._ROLE_PATH, pth)
                child.setData(0, self._ROLE_KIND, "file")
                child.setData(0, self._ROLE_SERIES_KEY, key)
                if is_corrupt:
                    # Grey the text + tooltip so it reads as disabled.
                    child.setForeground(0, QtGui.QColor("#b04646"))
                    child.setToolTip(
                        0, f"{pth}\n\nThis file is full-size on disk but "
                           f"its contents are all-null (0x00) — typically "
                           f"an aborted acquisition or interrupted copy. "
                           f"Unchecked by default so it won't break the "
                           f"batch.")
                parent.addChild(child)
            self.tree_batch_files.addTopLevelItem(parent)
            # Single-file series collapse — no point expanding a one-row group,
            # but expand a dead one so the ⚠ child is visible.
            parent.setExpanded(n > 1 or series_dead)

        self.tree_batch_files.blockSignals(False)
        self.tree_batch_files.itemChanged.connect(self._on_tree_item_changed)
        self._batch_update_summary()
        # Mark series + files that already have a saved polygon ROI
        self._refresh_batch_roi_markers()

    def _batch_iter_series(self):
        """Yield each top-level (series) item in the batch tree."""
        if not hasattr(self, "tree_batch_files"):
            return
        for i in range(self.tree_batch_files.topLevelItemCount()):
            yield self.tree_batch_files.topLevelItem(i)

    def _batch_iter_files(self):
        """Yield every (series_item, file_item) pair in the batch tree."""
        for ser in self._batch_iter_series():
            for j in range(ser.childCount()):
                yield ser, ser.child(j)

    def _batch_update_summary(self):
        n_series     = sum(1 for _ in self._batch_iter_series())
        n_sel_series = sum(1 for s in self._batch_iter_series()
                           if s.checkState(0) != Qt.CheckState.Unchecked)
        n_total_files = sum(1 for _ in self._batch_iter_files())
        n_sel_files   = sum(1 for _, c in self._batch_iter_files()
                            if c.checkState(0) == Qt.CheckState.Checked)
        if n_total_files == n_series:
            self.lbl_batch_summary.setText(
                f"{n_series} series / {n_sel_series} selected")
        else:
            self.lbl_batch_summary.setText(
                f"{n_series} series ({n_total_files} files) / "
                f"{n_sel_series} series selected ({n_sel_files} files)")
        folder = self.e_batch_folder.text().strip()
        if folder:
            self.lbl_batch_output_path.setText(
                f"Output → {os.path.join(folder, 'batch_results')}/<stem>/")
        else:
            self.lbl_batch_output_path.setText(
                "Output → (pick an input folder first)")

    def _batch_checked_series(self) -> "list[dict]":
        """Return one entry per series the user wants processed, each a
        dict {primary, key, files}: the primary file path (for stem +
        outdir naming), the series key, and the explicit list of
        checked sister files (in display order)."""
        out: list[dict] = []
        for s in self._batch_iter_series():
            if s.checkState(0) == Qt.CheckState.Unchecked:
                continue
            files = [s.child(j).data(0, self._ROLE_PATH)
                     for j in range(s.childCount())
                     if s.child(j).checkState(0)
                     == Qt.CheckState.Checked]
            if not files:
                continue
            out.append({
                "primary": s.data(0, self._ROLE_PATH),
                "key":     s.data(0, self._ROLE_SERIES_KEY),
                "files":   files,
            })
        return out

    def _batch_checked_files(self) -> list[str]:
        """Backwards-compatible flat list of primary paths for any series
        that has at least one checked file.  Kept for callers that only
        need to count what's selected."""
        return [g["primary"] for g in self._batch_checked_series()]

    # ── Batch queue (stack multiple jobs with their captured settings) ────────
    def _on_batch_add_to_queue(self):
        """Snapshot the current batch selection + sidebar settings as a queued
        job.  Because the params capture the live widget values now, changing
        the folder / preset afterwards and adding again gives a second job with
        its own settings — that's the whole point of the queue."""
        params_list = self._collect_batch_params()
        if not params_list:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to queue",
                "Pick a folder and check at least one file first "
                "(Import tab, Batch mode).")
            return
        folder = (os.path.basename(self.e_batch_folder.text().strip()
                                   .rstrip("/\\")) or "(folder)")
        preset = ""
        try:
            preset = self.c_preset.currentText()
        except Exception:
            pass
        label = f"{folder} - {len(params_list)} run(s)"
        if preset and preset != "- Current settings -":
            label += f"  [{preset}]"
        if not hasattr(self, "_batch_queue"):
            self._batch_queue = []
        self._batch_queue.append({"label": label, "params": params_list})
        self._refresh_batch_queue()

    def _refresh_batch_queue(self):
        if not hasattr(self, "lst_batch_queue"):
            return
        q = getattr(self, "_batch_queue", [])
        self.lst_batch_queue.clear()
        total = 0
        for j in q:
            self.lst_batch_queue.addItem(j["label"])
            total += len(j["params"])
        self.lbl_batch_queue.setText(f"Queue: {len(q)} job(s), {total} run(s)")
        for b in ("btn_batch_run_queue", "btn_batch_clear_queue",
                  "btn_batch_remove_queue"):
            w = getattr(self, b, None)
            if w is not None:
                w.setEnabled(len(q) > 0)

    def _on_batch_clear_queue(self):
        self._batch_queue = []
        self._refresh_batch_queue()

    def _on_batch_remove_queued(self):
        row = self.lst_batch_queue.currentRow()
        q = getattr(self, "_batch_queue", [])
        if 0 <= row < len(q):
            q.pop(row)
            self._refresh_batch_queue()

    def _on_batch_run_queue(self):
        """Run every queued job back-to-back: concatenate their params lists
        (each carries its own captured settings + output folder) and hand the
        flat list to the batch worker, which already processes runs in
        sequence."""
        q = getattr(self, "_batch_queue", [])
        if not q:
            QtWidgets.QMessageBox.warning(
                self, "Empty queue",
                "Add at least one job to the queue first.")
            return
        params_list = []
        for j in q:
            params_list.extend(j["params"])
        # Clear the queue only if the worker actually started — otherwise a
        # failed backend check / empty list would silently lose the queue.
        if self._launch_batch(params_list):
            self._batch_queue = []
            self._refresh_batch_queue()
