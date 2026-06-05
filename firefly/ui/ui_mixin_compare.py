"""MainWindow CompareMixin methods, split out of app_qt.py (#7)."""
from __future__ import annotations
from firefly.ui.ui_widgets import _CompareGroupCard

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


class CompareMixin:
    def _cmp_add_group(self):
        if len(self._cmp_group_cards) >= self.COMPARE_MAX_GROUPS:
            QtWidgets.QMessageBox.information(
                self, "Max groups reached",
                f"At most {self.COMPARE_MAX_GROUPS} groups can be compared "
                "at once.")
            return
        idx = len(self._cmp_group_cards)
        card = _CompareGroupCard(idx)
        card.delete_requested.connect(self._cmp_remove_group)
        # Live-update the Compare tab's "Analysis Configuration" wizard
        # (design summary / test plan / recommendation) whenever this card's
        # label, time point or folder set changes.  `_refresh_stats_preview`
        # is guarded (returns early until the centre wizard exists), so the
        # build-time seed calls are safe no-ops.
        card.changed.connect(self._refresh_stats_preview)
        self._cmp_groups_layout.addWidget(card)
        self._cmp_group_cards.append(card)
        self._refresh_stats_preview()

    def _cmp_remove_group(self, card: _CompareGroupCard):
        if len(self._cmp_group_cards) <= 2:
            QtWidgets.QMessageBox.information(
                self, "Minimum groups",
                "Need at least 2 groups for a comparison.")
            return
        self._cmp_group_cards.remove(card)
        self._cmp_groups_layout.removeWidget(card)
        card.deleteLater()
        # Re-number remaining cards
        for i, c in enumerate(self._cmp_group_cards):
            c.setTitle(f"Group {i + 1}")
        self._refresh_stats_preview()

    def _cmp_collect_groups(self) -> list[dict]:
        """Return the groups list in the shape `compare_groups` expects."""
        return [card.get_state() for card in self._cmp_group_cards]
