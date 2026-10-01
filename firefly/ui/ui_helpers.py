"""Icon factories, the motion colormap and small GUI utilities (open-folder, Qt
message handler).  Shared helpers used by the QML controllers and viewers.
"""
from __future__ import annotations

from firefly.analysis.fa_constants import MOTION_CLASS_COLORS, MOTION_CLASS_ORDER


_NAPARI_WELCOME_PHRASES = (
    "Drag image",
    "open image",
    "key bindings",
    "menu shortcuts",
    "Use the menu",
)


# Canonical motion-class colours/order live in fa_constants so the napari
# overlay, the single-run figure and the comparison figure can never drift
# apart.  (This palette is the reference scheme the others were standardised on.)
_MOTION_PALETTE = dict(MOTION_CLASS_COLORS)


_MOTION_ORDER = list(MOTION_CLASS_ORDER) + ["Unknown"]


_MOTION_CMAP_NAME = "firefly_motion"

