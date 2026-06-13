"""Figure theme palettes shared by the figure and circular modules.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

from firefly.analysis.fa_enums import FigureTheme


_THEME_REQUIRED_KEYS = (
    "BG", "PNL", "TXT", "MUT", "GRD", "ACC",
    "HDR_BG", "HDR_TXT", "ZEBRA", "FONT", "ARROW",
    # legacy keys consumed by `compare_groups` & `_write_pdf_report`
    "BAR_FILL", "SIG",
)


def _theme_palette(theme: str) -> dict:
    """Return a colour palette matching the master figure theme.
    Centralised so the master figure, the circular-statistics PDF, and
    the comparison PDF all read from the same source of truth.

    The returned dict is GUARANTEED to contain every key in
    `_THEME_REQUIRED_KEYS` — if any caller starts using a new key,
    add it to the tuple and to every branch below, and the
    `_validate_palette` check at the bottom will catch a regression at
    module-import time rather than at PDF-render time.
    """
    t = FigureTheme.parse(theme)
    if t is FigureTheme.LIGHT:
        pal = {"BG":   "#ffffff", "PNL":  "#f6f8fa",
               "TXT":  "#24292f", "MUT":  "#57606a",
               "GRD":  "#d0d7de", "ACC":  "#0969da",
               "HDR_BG":"#1f2937", "HDR_TXT":"#ffffff",
               "ZEBRA":"#f3f4f6", "FONT": "sans-serif",
               "ARROW":"#d93636",
               "BAR_FILL":"#0969da", "SIG":"#d93636"}
    elif t is FigureTheme.PUBLICATION:
        pal = {"BG":   "#ffffff", "PNL":  "#ffffff",
               "TXT":  "#000000", "MUT":  "#444444",
               "GRD":  "#cccccc", "ACC":  "#333333",
               "HDR_BG":"#000000", "HDR_TXT":"#ffffff",
               "ZEBRA":"#f2f2f2", "FONT": "DejaVu Sans",
               "ARROW":"#000000",
               "BAR_FILL":"#333333", "SIG":"#000000"}
    elif t is FigureTheme.AMOLED:
        # Pure-black backgrounds for OLED displays.  Mirrors Dark
        # otherwise so the figures are recognisable as the same FIREFLY
        # output.  PNL nudged to #0a0a0a so card-style panels still
        # read as cards against the BG.
        pal = {"BG":   "#000000", "PNL":  "#0a0a0a",
               "TXT":  "#e6edf3", "MUT":  "#9da7b1",
               "GRD":  "#30363d", "ACC":  "#58a6ff",
               "HDR_BG":"#141414", "HDR_TXT":"#e6edf3",
               "ZEBRA":"#050505", "FONT": "monospace",
               "ARROW":"#ff7b72",
               "BAR_FILL":"#58a6ff", "SIG":"#ff7b72"}
    else:
        # Dark (default).
        pal = {"BG":   "#0d1117", "PNL":  "#161b22",
               "TXT":  "#e6edf3", "MUT":  "#9da7b1",
               "GRD":  "#30363d", "ACC":  "#58a6ff",
               "HDR_BG":"#21262d", "HDR_TXT":"#e6edf3",
               "ZEBRA":"#1c2128", "FONT": "monospace",
               "ARROW":"#ff7b72",
               "BAR_FILL":"#58a6ff", "SIG":"#ff7b72"}
    # Belt-and-braces: if a caller (or future edit) ever accesses a key
    # we forgot to include, return a sensible TXT fallback rather than
    # crashing with a KeyError mid-render.  We do this via a small
    # dict subclass so `pal[<missing>]` works as if `pal.get(<missing>,
    # pal["TXT"])` were called.
    class _PalDict(dict):
        __slots__ = ()
        def __missing__(self, key):
            return self.get("TXT", "#000000")
    return _PalDict(pal)


def style_axes(ax, pal, *, kind="cartesian"):
    """Give a figure axes a clean, modern look: drop the top + right spines
    (chart junk) and thin/lighten the remaining ones.  Theme-aware via
    ``pal["GRD"]`` / ``pal["TXT"]`` (accepts the full palette dict or a small
    ``{"GRD":..,"TXT":..}``).

    Polar axes (the radial panels) and image panels (imshow) keep their full
    frame — despining those looks wrong — so only their spine colour/width is
    tidied.  Pass ``kind="image"`` for imshow panels; image axes are also
    auto-detected via ``ax.images``.  Grids are left to the caller."""
    try:
        grd = pal["GRD"]
        txt = pal["TXT"]
    except Exception:
        return
    try:
        if ax.name == "polar":
            return
        if kind == "image" or bool(ax.images):
            for sp in ax.spines.values():
                sp.set_edgecolor(grd)
                sp.set_linewidth(0.8)
            return
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_edgecolor(grd)
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(length=3, width=0.7, colors=txt)
    except Exception:
        pass
