"""Regenerate the Figures-preview sample panels for each figure theme.

    python tools/gen_preview_panels.py

Writes firefly/ui/qml/assets/figures/panel_A_{dark,light,publication}.png so the
Preferences ▸ Figures live preview reflects the chosen theme instead of showing a
single dark render on every theme.  The projection is synthetic but representative
(a curved filament of emitters) and is generated ONCE for all three, so switching
themes re-themes the *same* cell — exactly what the preview is meant to show.  The
theme palettes mirror fa_figure.py so the preview matches real output.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

# BG / TXT mirror fa_figure.py's Dark / Light / Publication palettes.
THEMES = {
    "dark":        dict(BG="#0d1117", TXT="#e6edf3", font="monospace"),
    "light":       dict(BG="#ffffff", TXT="#24292f", font="sans-serif"),
    "publication": dict(BG="#ffffff", TXT="#000000", font="DejaVu Sans"),
}
PX_UM = 0.15873
DEST = os.path.join("firefly", "ui", "qml", "assets", "figures")


def make_projection(n=256, seed=7):
    """A faint background + a curved filament of gaussian emitters — looks like a
    real sptPALM max-projection.  Fixed seed → reproducible."""
    rng = np.random.default_rng(seed)
    # a visible purple inferno haze (scattered background emitters + noise) so it
    # reads like a real projection, not a clean line on black
    img = rng.gamma(1.1, 0.05, size=(n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    for _ in range(900):                       # diffuse background localisations
        x0, y0 = rng.uniform(0, n), rng.uniform(0, n)
        img += rng.uniform(0.05, 0.18) * np.exp(
            -((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * rng.uniform(2, 4) ** 2))
    t = np.linspace(0, 1, 260)                 # the bright curved filament
    cx = 40 + 150 * t
    cy = 210 - 170 * t + 28 * np.sin(t * 3.0)
    for x0, y0 in zip(cx, cy):
        amp = rng.uniform(0.4, 1.2)
        sig = rng.uniform(1.8, 3.4)
        img += amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig ** 2))
    img += rng.normal(0, 0.02, img.shape).clip(0)
    return img


def render(theme, img, out):
    th = THEMES[theme]
    plt.rcParams.update({"font.family": th["font"]})
    fig = plt.figure(figsize=(3.0, 3.0), dpi=150, facecolor=th["BG"])
    ax = fig.add_axes([0.16, 0.13, 0.80, 0.78])
    ax.set_facecolor(th["BG"])
    ax.imshow(img, cmap="inferno", origin="lower", extent=[0, 256, 0, 256],
              vmax=np.percentile(img, 99.3))
    ax.set_title("Max Projection", color=th["TXT"], fontsize=9, loc="left",
                 fontweight="bold")
    ax.set_xlabel("X (%.5g um/px)" % PX_UM, color=th["TXT"], fontsize=7)
    ax.set_ylabel("Y (px)", color=th["TXT"], fontsize=7)
    ax.tick_params(colors=th["TXT"], labelsize=6)
    for s in ax.spines.values():
        s.set_color(th["TXT"])
    bar_px = 5.0 / PX_UM
    ax.plot([20, 20 + bar_px], [18, 18], color="white", lw=2)
    ax.text(20, 26, "5 µm", color="white", fontsize=6)
    fig.savefig(out, facecolor=th["BG"], dpi=150)
    plt.close(fig)


def main():
    img = make_projection()
    os.makedirs(DEST, exist_ok=True)
    for theme in THEMES:
        out = os.path.join(DEST, "panel_A_%s.png" % theme)
        render(theme, img, out)
        print("wrote", out)


if __name__ == "__main__":
    main()
