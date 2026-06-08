"""MainWindow HandlersMixin methods, split out of app_qt.py (#7)."""
from __future__ import annotations
import queue

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar

from firefly import sptpalm_analysis
from firefly import crash_reporter
from firefly import cuda_installer
from firefly.ui.ui_theme import _THEME
from firefly.ui.ui_constants import (TAB_IMPORT, TAB_ANALYSIS, TAB_COMPARE,
                          TAB_RESULTS, TAB_VISUALISE, TAB_REPROCESS)
from firefly.ui.ui_helpers import (_make_cogwheel_icon, _make_close_x_icon,
                        _make_napari_container_layout_opaque, _hide_napari_chrome,
                        _register_motion_colormap, _open_folder,
                        _MOTION_PALETTE, _MOTION_ORDER, _MOTION_CMAP_NAME)
from firefly.ui.ui_widgets import (_UpdateCheckThread, _UpdateDialog, _ModeTile, _ActionTile, _QuietSpinBox,
                        _QuietDoubleSpinBox, _QuietComboBox, _CollapsibleSection,
                        _ResourceMonitor, _MassHistogram, _LiveFrameView,
                        _TrackInspector, _ResultsPanel, _RoiDialog, _RoiViewer,
                        _FolderDropList, _CompareGroupCard, _PreferencesDialog,
                        _load_imagej_roi_polygons, _load_tif_mask_polygons,
                        _load_any_roi_file)


class HandlersMixin:
    def _current_version(self) -> str:
        try:
            from firefly import sptpalm_analysis as _sa
            return str(getattr(_sa, "__version__", "0.0.0"))
        except Exception:
            return "0.0.0"

    def _on_update_available(self, latest_tag: str, release):
        """Slot called when the startup check finds a newer release.  Stash
        the full release dict and light the header pill; clicking it opens
        the in-app update dialog (download + install + relaunch)."""
        if not hasattr(self, "btn_update_pill"):
            return
        # Respect a "skip this version" choice from a previous session — but
        # still surface anything strictly newer than the skipped tag.
        try:
            from firefly import updater
            skip = QtCore.QSettings("jacoblevers", "FIREFLY").value(
                "updates/skip_version", "") or ""
            if skip and (updater.parse_version(latest_tag)
                         <= updater.parse_version(str(skip))):
                return
        except Exception:
            pass
        self._latest_release = release if isinstance(release, dict) else {}
        self._update_url = (self._latest_release.get("html_url")
                            or self._UPDATE_RELEASES_URL)
        self.btn_update_pill.setText(f"  ●  Update available: {latest_tag}  ")
        self.btn_update_pill.setToolTip(
            f"FIREFLY {latest_tag} is available.  Click to update.")
        self.btn_update_pill.setVisible(True)

    def _on_update_pill_clicked(self):
        release = getattr(self, "_latest_release", None)
        dlg = _UpdateDialog(self, self._current_version(), release)
        dlg.exec()

    def _force_check_for_updates(self):
        """User-triggered check (menu / Preferences).  Always opens the
        update dialog with the result — even when up to date or offline."""
        thread = _UpdateCheckThread(self._UPDATE_API_URL,
                                    self._current_version(),
                                    parent=self, force=True)
        self._force_update_thread = thread
        try:
            self.statusBar().showMessage("Checking for updates…", 3000)
        except Exception:
            pass
        thread.check_finished.connect(self._on_force_check_finished)
        thread.start()

    def _on_force_check_finished(self, release):
        try:
            self.statusBar().clearMessage()
        except Exception:
            pass
        try:
            t = getattr(self, "_force_update_thread", None)
            if t is not None:
                t.quit(); t.wait(1000)
        except Exception:
            pass
        self._force_update_thread = None
        dlg = _UpdateDialog(self, self._current_version(), release)
        dlg.exec()

    def _apply_downloaded_update(self, path) -> bool:
        """Install a downloaded update and quit so the staged helper can
        swap files + relaunch.  Returns True if the swap was initiated (the
        app is quitting); False if it couldn't be applied (an error box was
        shown and the download folder revealed)."""
        from firefly import updater
        if not updater.is_frozen():
            QtWidgets.QMessageBox.information(
                self, "Update downloaded",
                "The update was downloaded, but in-app install only works in "
                "the packaged FIREFLY app. Use 'git pull' for a source "
                "install.")
            return False
        try:
            updater.apply_update(path)
        except updater.UpdaterError as exc:
            box = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Icon.Warning,
                "Couldn't install update", str(exc),
                QtWidgets.QMessageBox.StandardButton.Ok, self)
            box.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            box.exec()
            reveal = getattr(exc, "reveal_path", None)
            if reveal:
                try:
                    _open_folder(reveal)
                except Exception:
                    pass
            return False
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Couldn't install update",
                                          str(exc))
            return False
        # Success — quit (deferred so the modal update dialog can close
        # first) so the detached helper can replace the app + relaunch.
        QtCore.QTimer.singleShot(0, self.close)
        return True

    def _on_console_visibility(self, visible: bool):
        """Keep the status-bar toggle button's checked state in sync."""
        try:
            self.btn_show_console.setChecked(visible)
        except AttributeError:
            pass

    def _on_console_dock_moved(self, area):
        """Give the console dock a usable size whenever Qt re-docks it.

        Bottom-docked: ~200 px tall so ~12 lines of log are visible.
        Right-docked:  ~420 px wide so most log lines don't wrap.
        Without this, Qt's default for a fresh right-dock is a ~40-px
        strip — the regression the user reported.
        """
        try:
            if area == Qt.DockWidgetArea.RightDockWidgetArea \
               or area == Qt.DockWidgetArea.LeftDockWidgetArea:
                self.resizeDocks([self._console_dock], [420],
                                 Qt.Orientation.Horizontal)
            elif area == Qt.DockWidgetArea.BottomDockWidgetArea \
                 or area == Qt.DockWidgetArea.TopDockWidgetArea:
                self.resizeDocks([self._console_dock], [200],
                                 Qt.Orientation.Vertical)
        except Exception:
            pass

    def _on_tab_changed_swap_sidebar(self, idx: int):
        """Swap both sidebar stacks (settings + bottom button) to match
        the active tab, and re-label the sidebar header.

        Tab order is fixed by `__init__`: 0=Import, 1=Analysis,
        2=Compare, 3=Visualise, 4=Re-process.  Sidebar pages share those
        indices.  Per the user spec the Analysis tab reuses the Import
        sidebar wholesale — there's no separate page for it; the handler
        just routes idx=1 → page 0.  (The former standalone Statistics
        tab was merged into Compare, so there's no page 3=Statistics.)
        """
        # Defensive clamp — Qt may emit currentChanged during shutdown.
        if idx < 0 or idx >= self.tabs.count():
            return
        # Analysis tab piggybacks Import's sidebar.
        page_idx = 0 if idx == 1 else idx
        try:
            self._sidebar_stack.setCurrentIndex(
                min(page_idx, self._sidebar_stack.count() - 1))
            self._sidebar_action.setCurrentIndex(
                min(page_idx, self._sidebar_action.count() - 1))
        except Exception:
            return
        titles = {
            0: "Analysis Parameters",
            1: "Analysis Parameters",   # Analysis mirrors Import
            2: "Comparison",
            3: "Results",
            4: "Visualise",
            5: "Re-process",
        }
        try:
            self._sidebar_title.setText(titles.get(idx, ""))
        except Exception:
            pass

    def _open_previous_comparison(self):
        """Load a saved comparison's results file into the Results tab."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open a previous comparison",
            "", "Comparison results (*_results.json);;All files (*)")
        if not path:
            return
        if not path.endswith("_results.json"):
            QtWidgets.QMessageBox.information(
                self, "No results file",
                "That doesn't look like a FIREFLY results file. Pick the "
                "'*_results.json' written next to a comparison's outputs "
                "(comparisons run before v2.46 don't have one — just re-run "
                "the comparison to generate it).")
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._results_view.load(data, base_dir=os.path.dirname(path))
            self._switch_to_tab(TAB_RESULTS)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Couldn't open results",
                f"Failed to read the results file:\n{exc}")

    def _on_import_mode_changed(self, mode):
        """Show whichever sub-panel matches the new mode and hide the
        others.  Accepts a string ("single" / "batch" / "csv") for the
        new tri-state mode toggle; falls back to bool for the legacy
        two-mode call sites."""
        # Legacy callers pass a bool (single=True / batch=False).  Any stale
        # "csv" value (from before External CSV was folded into single mode)
        # collapses to "single".
        if isinstance(mode, bool):
            mode = "single" if mode else "batch"
        if mode == "csv":
            mode = "single"
        self._import_mode = mode
        try:    self._single_panel.setVisible(mode == "single")
        except AttributeError: pass
        try:    self._batch_panel.setVisible(mode == "batch")
        except AttributeError: pass

    def _on_batch_pick_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder containing input files",
            self.e_batch_folder.text() or os.path.expanduser("~"))
        if path:
            self.e_batch_folder.setText(path)
            self._batch_rescan(path)

    def _on_batch_rescan(self):
        path = self.e_batch_folder.text().strip()
        if path:
            self._batch_rescan(path)

    def _on_tree_item_changed(self, item: "QtWidgets.QTreeWidgetItem",
                              _col: int):
        """Keep parent ↔ child check states in sync, then refresh the
        summary.  Re-entrancy-guarded because every setCheckState we
        make in here would otherwise re-fire this slot."""
        if self._tree_propagation_guard:
            return
        self._tree_propagation_guard = True
        try:
            kind = item.data(0, self._ROLE_KIND)
            if kind == "series":
                # Push the parent's new state down to children
                state = item.checkState(0)
                if state != Qt.CheckState.PartiallyChecked:
                    for j in range(item.childCount()):
                        item.child(j).setCheckState(0, state)
            elif kind == "file":
                parent = item.parent()
                if parent is not None:
                    n  = parent.childCount()
                    on = sum(1 for j in range(n)
                             if parent.child(j).checkState(0)
                             == Qt.CheckState.Checked)
                    if on == 0:
                        parent.setCheckState(0, Qt.CheckState.Unchecked)
                    elif on == n:
                        parent.setCheckState(0, Qt.CheckState.Checked)
                    else:
                        parent.setCheckState(0, Qt.CheckState.PartiallyChecked)
        finally:
            self._tree_propagation_guard = False
            self._batch_update_summary()

    def _on_batch_tree_item_double_clicked(
            self, item: "QtWidgets.QTreeWidgetItem", _col: int):
        """Double-click on a tree row loads it into the preview viewer.
        Single-click only highlights — checkbox toggles are cheap and
        the heavy file load happens exclusively from here or the
        toolbar button."""
        if item is None:
            return
        path = item.data(0, self._ROLE_PATH)
        if path:
            self._roi_load_specific_path(path)

    def _on_batch_open_in_viewer(self):
        """Load the currently-highlighted tree item into the preview
        viewer.  Triggered by the "Open in viewer" toolbar button."""
        if not hasattr(self, "tree_batch_files"):
            return
        it = self.tree_batch_files.currentItem()
        if it is None:
            QtWidgets.QMessageBox.information(
                self, "Open in viewer",
                "Click a series or file in the tree first to highlight "
                "it, then press 'Open in viewer'.")
            return
        path = it.data(0, self._ROLE_PATH)
        if path:
            self._roi_load_specific_path(path)

    def _on_batch_select_all(self):
        for s in self._batch_iter_series():
            s.setCheckState(0, Qt.CheckState.Checked)

    def _on_batch_select_none(self):
        for s in self._batch_iter_series():
            s.setCheckState(0, Qt.CheckState.Unchecked)

    def _on_batch_select_inverse(self):
        for s in self._batch_iter_series():
            cur = s.checkState(0)
            s.setCheckState(0,
                Qt.CheckState.Unchecked
                if cur == Qt.CheckState.Checked
                else Qt.CheckState.Checked)

    def _on_roi_polygons_changed(self, file_path: str, polys: list):
        """The embedded viewer emits this whenever the user adds/edits/
        removes a polygon.  Auto-persist to QSettings."""
        if not file_path:
            return
        key = os.path.abspath(file_path)
        if polys:
            self._roi_polygons[key] = polys
        else:
            self._roi_polygons.pop(key, None)
        self._save_roi_polygons()
        # Refresh status indicators
        self._refresh_single_roi_status()
        self._refresh_batch_roi_markers()

    def _on_roi_mode_changed(self, text: str):
        """Grey out threshold/projection controls that don't apply to the
        active mode and push the mask overlay to the viewer."""
        is_auto    = text == "Auto threshold"
        is_manual  = text == "Manual threshold"

        # Auto method only meaningful when in Auto-threshold mode
        try: self.c_roi_auto_method.setEnabled(is_auto)
        except AttributeError: pass
        # Manual threshold spinbox only used in Manual-threshold mode
        try: self.s_roi_threshold.setEnabled(is_manual)
        except AttributeError: pass
        try: self.sld_roi_threshold.setEnabled(is_manual)
        except AttributeError: pass
        # Projection (mean vs sum) is irrelevant in polygon mode (we use
        # a sample mean for the preview regardless) and in None mode.
        try: self.c_roi_mask_mode.setEnabled(is_auto or is_manual)
        except AttributeError: pass

        self._push_roi_mask_params()

    def _on_single_set_roi(self):
        path = self.e_file.text().strip()
        if not path or not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(
                self, "ROI editor",
                "Pick an input file first on the Import tab.")
            return
        if self._open_roi_dialog(path):
            self._refresh_single_roi_status()

    def _on_batch_set_roi(self):
        # Use the highlighted item (currentItem), not the checked ones,
        # so the user can edit ROIs without changing what's selected
        # for processing.
        it = self.tree_batch_files.currentItem() \
            if hasattr(self, "tree_batch_files") else None
        if it is None:
            QtWidgets.QMessageBox.warning(
                self, "ROI editor",
                "Click a file or series in the tree to highlight it, "
                "then click Set ROI… again.")
            return
        path = it.data(0, self._ROLE_PATH)
        if self._open_roi_dialog(path):
            self._refresh_batch_roi_markers()

    def _on_batch_clear_roi(self):
        it = self.tree_batch_files.currentItem() \
            if hasattr(self, "tree_batch_files") else None
        if it is None:
            return
        path = it.data(0, self._ROLE_PATH)
        key = os.path.abspath(path) if path else None
        if key and key in self._roi_polygons:
            del self._roi_polygons[key]
            self._save_roi_polygons()
            self._refresh_batch_roi_markers()

    def _on_cmp_browse_outdir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose comparison output folder",
            self.e_cmp_outdir.text() or os.path.expanduser("~"))
        if path:
            self.e_cmp_outdir.setText(path)

    def _on_browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select input file", os.path.expanduser("~"),
            "Images or localisations "
            "(*.czi *.tif *.tiff *.csv *.txt *.tsv);;"
            "Image stacks (*.czi *.tif *.tiff);;"
            "Localisations (*.csv *.txt *.tsv);;"
            "All files (*)")
        if path:
            self.e_file.setText(path)
            if not self.e_outdir.text():
                self.e_outdir.setText(os.path.dirname(path))

    def _on_browse_outdir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder", os.path.expanduser("~"))
        if path:
            self.e_outdir.setText(path)

    def _on_browse_csv_bg(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select background image (optional)",
            self.e_outdir.text() or os.path.dirname(self.e_file.text())
            or os.path.expanduser("~"),
            "Image stacks (*.czi *.tif *.tiff);;All files (*)")
        if path:
            self.e_csv_bg.setText(path)

    def _on_load_manifest(self):
        """Open a `<stem>_run_manifest.json` and apply its widget_state
        snapshot to the sidebar, plus repopulate the input/output paths."""
        import json
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open run manifest",
            self.e_outdir.text() or os.path.expanduser("~"),
            "Manifest (*_run_manifest.json);;JSON (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Couldn't load manifest", str(exc))
            return
        # Apply widget snapshot (most important)
        state = manifest.get("widget_state") or {}
        self._apply_widget_state(state)
        # Path fields: try the original input path; if missing, leave alone
        inp = (manifest.get("input") or {}).get("path", "") or ""
        if inp and os.path.isfile(inp):
            self.e_file.setText(inp)
        # Output folder
        outd = manifest.get("output_dir", "")
        if outd:
            self.e_outdir.setText(outd)
        # Status feedback
        v = manifest.get("firefly_version", "?")
        when = manifest.get("created_at", "?")
        self.statusBar().showMessage(
            f"Loaded manifest from {os.path.basename(path)}  "
            f"(FIREFLY {v}, {when})", 8000)

    def _on_preset_picked(self, name: str) -> None:
        """Apply a preset to the sidebar when the user picks one from the
        combobox.  Ignores the leading '— Current settings —' sentinel."""
        if not name or name.startswith("—"):
            # "— Current settings —" sentinel: no preset baseline → never
            # "modified".
            self._active_preset_state = None
            try:    self._refresh_preset_modified()
            except Exception: pass
            return
        import json
        path = os.path.join(self._presets_dir(), f"{name}.json")
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Couldn't load preset", str(exc))
            return
        # Drop our own internal tag before applying
        state.pop(self._BUILTIN_PRESETS_TAG, None)
        # Suspend the modified-watch while applying (the per-widget signals
        # would otherwise transiently flag "modified" mid-apply), then capture
        # the ACTUAL resulting state as the baseline so a freshly-applied preset
        # reads as unmodified.
        self._suspend_modified_watch = True
        try:
            self._apply_widget_state(state)
        finally:
            self._suspend_modified_watch = False
        try:
            self._active_preset_state = self._widget_state_dict()
            self._refresh_preset_modified()
        except Exception:
            pass
        self.statusBar().showMessage(f"Applied preset: {name}", 5000)

    def _on_preset_delete(self) -> None:
        """Remove the currently-selected preset from disk after
        confirmation."""
        name = self.c_preset.currentText() if hasattr(self, "c_preset") else ""
        if not name or name.startswith("—"):
            QtWidgets.QMessageBox.information(
                self, "No preset selected",
                "Pick a preset from the dropdown first, then click Delete.")
            return
        path = os.path.join(self._presets_dir(), f"{name}.json")
        if not os.path.isfile(path):
            return
        # Heads-up if the user is about to delete a built-in: it'll come
        # back on next launch from the seeding logic.
        import json
        is_builtin = False
        try:
            with open(path, "r", encoding="utf-8") as fh:
                is_builtin = bool(json.load(fh).get(
                    self._BUILTIN_PRESETS_TAG, False))
        except Exception:
            pass
        msg = f"Delete preset '{name}'?"
        if is_builtin:
            msg += ("\n\nThis is a built-in preset — it will be re-created "
                    "on the next FIREFLY launch unless you save your own "
                    "version with the same name first.")
        ret = QtWidgets.QMessageBox.question(
            self, "Delete preset", msg,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No)
        if ret != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:    os.remove(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Delete failed", str(exc))
            return
        self._refresh_preset_combo()
        # Park selection on the "current settings" sentinel
        try:    self.c_preset.setCurrentIndex(0)
        except Exception: pass
        self.statusBar().showMessage(f"Deleted preset: {name}", 5000)

    def _on_preset_save(self) -> None:
        """Prompt for a name and write the current sidebar to disk."""
        import json, re
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save preset",
            "Name this preset (use letters, numbers, spaces, '-' or '_'):")
        if not ok:
            return
        name = (name or "").strip()
        if not name:
            return
        # Sanitise the filename — no path separators, control chars, etc.
        if not re.match(r"^[A-Za-z0-9 _\-]+$", name):
            QtWidgets.QMessageBox.warning(
                self, "Invalid name",
                "Preset names can only contain letters, numbers, "
                "spaces, '-' and '_'.")
            return
        path = os.path.join(self._presets_dir(), f"{name}.json")
        if os.path.isfile(path):
            ret = QtWidgets.QMessageBox.question(
                self, "Overwrite preset?",
                f"'{name}' already exists.  Overwrite?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No)
            if ret != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        state = self._widget_state_dict()
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Save failed", str(exc))
            return
        self._refresh_preset_combo()
        # Surface the new preset as the current selection
        try:
            self.c_preset.setCurrentText(name)
        except Exception:
            pass
        self.statusBar().showMessage(f"Saved preset: {name}", 5000)

    def _on_run_clicked(self):
        # Acting as Stop?
        if self._proc is not None and self._proc.is_alive():
            if self._cancel_event is not None:
                self._cancel_event.set()
            # Record when Stop was requested so the poller can escalate
            # (SIGTERM → SIGKILL) if cooperative cancel doesn't take
            # effect within a few seconds.  Without this, a user clicking
            # Stop during a long uninterruptible region (e.g. trackpy's
            # linker on a high-density chunk) sees nothing happen for
            # minutes.
            self._stop_requested_at = time.time()
            self._stop_escalation_stage = 0   # 0=cooperative, 1=SIGTERM, 2=SIGKILL
            self.btn_run.setText("Stopping…")
            self.btn_run.setEnabled(False)

            # Surface in the shared console + status bar so the user
            # knows their click registered (without forcing them to open
            # the console panel).
            self.console_log.appendPlainText(
                "\n── Stop requested.  Waiting for the current stage to reach "
                "a checkpoint (up to 5 s); will force-terminate if it doesn't.")
            self.statusBar().showMessage("Stop requested — waiting for current stage…")
            return

        # Dispatch.  The Compare-tab's own Generate button overrides; the
        # sidebar Start button uses the Import-tab mode (Single / Batch),
        # OR if the active tab is Compare, runs the comparison.
        sender = self.sender()
        if sender is getattr(self, "btn_cmp_run", None):
            self._start_compare_run()
            return
        active_tab_label = self.tabs.tabText(self.tabs.currentIndex())
        if active_tab_label.startswith(TAB_COMPARE):
            self._start_compare_run()
        elif self.r_mode_batch.isChecked():
            self._start_batch_run()
        else:
            self._start_single_run()

    def _on_poll_queue(self):
        """Drain pending messages from the subprocess's message queue.

        Called on a QTimer at ~30 Hz while a run is active.  We process at
        most a few hundred messages per tick to keep the UI responsive
        when tqdm is spamming progress updates during fast stages.
        """
        if self._msg_queue is None:
            return
        # Drain up to N messages per tick so we don't starve the UI loop
        # on a fast log flood.  Bumped to 1000 because most messages are
        # cheap log lines that we batch into a single appendPlainText.
        budget = 1000
        worker_done = False
        is_batch   = getattr(self, "_is_batch_run", False)
        is_compare = getattr(self, "_is_compare_run", False)

        # All log lines now land in the shared Console dock — one place
        # for everything.  The per-tab widgets are only the stage label
        # and the progress bar.
        log_widget = self.console_log
        if is_compare:
            progress_widget = self.cmp_progress
            stage_label     = self.cmp_stage_label
        elif is_batch:
            progress_widget = self.batch_progress
            stage_label     = self.batch_stage_label
        else:
            progress_widget = self.progress_bar
            stage_label     = self.run_stage_label

        # Buffer log lines and append them in a SINGLE call at end of tick.
        # appendPlainText reflows the document each call; 1000 separate
        # appends on a long document can take seconds.  One append of a
        # newline-joined string completes in milliseconds.
        log_buf: list[str] = []
        last_progress: tuple | None = None  # only the latest progress matters

        while budget > 0:
            try:
                kind, payload = self._msg_queue.get_nowait()
            except queue.Empty:
                break
            budget -= 1
            if kind == "log":
                log_buf.append(payload)
            elif kind == "progress":
                last_progress = payload   # drop earlier intra-tick updates
            elif kind == "mass_chunk":
                # Live histogram update from the localisation stream
                try:    self.mass_hist.add_chunk(payload)
                except AttributeError: pass
            elif kind == "preview_frame":
                # Live detection-view update.  Payload carries a flat
                # bytes blob + shape so we can reconstruct the frame
                # array without round-tripping through numpy in the
                # queue (lighter and works in subprocess-spawned land).
                try:
                    import numpy as _np
                    shape = payload.get("shape") or [0, 0]
                    blob  = payload.get("frame")
                    if blob and shape[0] and shape[1]:
                        arr = _np.frombuffer(blob, dtype=_np.float32) \
                                 .reshape(shape[0], shape[1])
                        self.live_view.set_frame(
                            arr,
                            payload.get("xs", []),
                            payload.get("ys", []),
                            payload.get("idx", 0),
                            payload.get("n_frames", 0))
                except (AttributeError, ValueError, KeyError):
                    pass
            elif kind == "done":
                # Single-file completion.  Only valid in non-batch mode;
                # in batch mode the per-file messages are "file_done".
                self._handle_done(payload)
                worker_done = True
            elif kind == "file_starting":
                # New file in a batch — wipe the mass histogram so it
                # doesn't accumulate values from the previous file's
                # localisations.  Live view is fine — preview_frame
                # messages naturally overwrite as they arrive.
                try:    self.mass_hist.reset()
                except AttributeError: pass
                # Restart the pipeline stage map for the new file.
                try:    self.pipeline_diagram.reset()
                except AttributeError: pass
                # Update the overall-batch bar (files remaining).
                try:    self._handle_file_starting(payload)
                except AttributeError: pass
            elif kind == "file_done":
                self._handle_file_done(payload)
            elif kind == "file_error":
                self._handle_file_error(payload)
            elif kind == "batch_done":
                self._handle_batch_done(payload)
                worker_done = True
            elif kind == "compare_done":
                self._handle_compare_done(payload)
                worker_done = True
            elif kind == "compare_error":
                self._handle_compare_error(payload)
                worker_done = True
            elif kind == "stopped":
                self._handle_stopped()
                worker_done = True
            elif kind == "error":
                self._handle_failed(payload)
                worker_done = True

        # Flush the per-tick log buffer with ONE append call.  Also
        # coalesce progress: only the most recent value matters for
        # display purposes (it overwrites all earlier ones anyway).
        if log_buf:
            log_widget.appendPlainText("\n".join(log_buf))
        if last_progress is not None:
            pct, msg = last_progress
            progress_widget.setValue(pct)
            # Show the current step IN the bar alongside the % so the bar is
            # self-describing (e.g. "Localising… — 95%").  Trim an over-long
            # message (e.g. a long file path) so it doesn't overflow the bar.
            _m = (msg or "").strip()
            if len(_m) > 48:
                _m = _m[:47] + "…"
            progress_widget.setFormat(f"{_m}  —  {pct}%" if _m else f"{pct}%")
            stage_label.setText(msg)
            self.statusBar().showMessage(msg)
            # Light up the Analysis cockpit's pipeline stage map (Analysis /
            # batch only — Compare messages don't map to pipeline stages).
            if not is_compare:
                try:    self.pipeline_diagram.set_stage_from_msg(msg)
                except AttributeError: pass

        # Stop-button escalation: if cancel_event was set N seconds ago
        # and the subprocess is still alive, escalate.  Two-stage SIGTERM
        # → SIGKILL because some torch / native code can ignore SIGTERM.
        stop_at = getattr(self, "_stop_requested_at", None)
        if (stop_at is not None and self._proc is not None
                and self._proc.is_alive()):
            elapsed = time.time() - stop_at
            stage   = getattr(self, "_stop_escalation_stage", 0)
            if stage == 0 and elapsed > 5.0:
                log_widget.appendPlainText(
                    "  Cooperative cancel didn't take effect within 5 s — "
                    "sending SIGTERM to the analysis subprocess.")
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                self._stop_escalation_stage = 1
                self._stop_requested_at = time.time()  # reset timer for SIGKILL
            elif stage == 1 and elapsed > 3.0:
                log_widget.appendPlainText(
                    "  SIGTERM didn't take effect within 3 s — sending SIGKILL.")
                try:
                    self._proc.kill()
                except Exception:
                    pass
                self._stop_escalation_stage = 2

        # Also detect a subprocess that has exited without posting a
        # terminal message (e.g. crashed, SIGTERM'd, or SIGKILL'd).
        # IMPORTANT: never time.sleep() in this slot — it runs on the
        # GUI thread on a 30 Hz QTimer.  Drain whatever's already in
        # the queue non-blockingly; any final logs the subprocess
        # wrote in the last instant will be picked up by the next
        # tick (~33 ms away) anyway.
        if not worker_done and self._proc is not None and not self._proc.is_alive():
            # Drain any pending logs without blocking.
            for _ in range(64):
                try:
                    kind, payload = self._msg_queue.get_nowait()
                except (queue.Empty, Exception):
                    break
                if kind == "log":
                    log_widget.appendPlainText(payload)
            # If the user pressed Stop, treat exit as "stopped", not an error
            if getattr(self, "_stop_requested_at", None) is not None:
                self._handle_stopped()
            else:
                self._handle_failed(
                    f"Analysis subprocess exited abnormally "
                    f"(exit code {self._proc.exitcode}).  See log for details.")
            worker_done = True

        if worker_done:
            self._cleanup_after_run()

    def _on_elapsed_tick(self):
        if self._run_start_time is None:
            return
        import time as _time
        try:
            self.lbl_elapsed.setText(
                f"Elapsed: {self._format_elapsed(_time.monotonic() - self._run_start_time)}")
        except AttributeError:
            pass

    def _on_cuda_button_clicked(self):
        """Manual entry point from the Performance section button."""
        try:
            from firefly import cuda_installer as _cu
        except Exception:
            QtWidgets.QMessageBox.warning(
                self, "CUDA installer unavailable",
                "The CUDA installer module could not be loaded.")
            return
        if _cu.is_installed():
            reply = QtWidgets.QMessageBox.question(
                self, "Remove CUDA acceleration?",
                "CUDA acceleration is currently installed at\n"
                f"{_cu.sidecar_dir()}\n\n"
                "Remove it?  (You can reinstall any time.)",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                try:
                    _cu.uninstall()
                    QtWidgets.QMessageBox.information(
                        self, "CUDA removed",
                        "CUDA acceleration has been removed.  Restart "
                        "FIREFLY to drop back to the bundled CPU build.")
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(
                        self, "Removal failed", str(exc))
            return
        # Not installed → kick off the same flow as the auto-prompt.
        try:
            _cu.clear_declined()
        except Exception:
            pass
        self._run_cuda_install()
