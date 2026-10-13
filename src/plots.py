"""Shared, consistent figure styling for the report and dashboard."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# fixed categorical order (never cycled); colour follows the entity
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CLASS_COLORS = {0: SERIES[0], 1: SERIES[1]}          # legit = blue, fraud = orange
MODEL_COLORS = {"Logistic Regression": SERIES[0], "Random Forest": SERIES[1],
                "XGBoost": SERIES[2], "MLP": SERIES[3], "LSTM": SERIES[4],
                "Rule baseline": "#8a8984"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.family": "DejaVu Sans", "figure.dpi": 110,
})


def save(fig, name):
    fig.tight_layout()
    path = C.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def hbar(series: pd.Series, title, xlabel, name, color=SERIES[0], highlight=None, fmt="{:.3f}"):
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, max(2.5, 0.32 * len(s) + 1)))
    cols = [SERIES[1] if highlight and highlight(k, v) else color for k, v in s.items()]
    ax.barh(s.index, s.values, color=cols, height=0.65, edgecolor=SURFACE, linewidth=1)
    for i, v in enumerate(s.values):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color=INK2)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    return save(fig, name)


def grouped_bar(df: pd.DataFrame, x, hue, y, title, ylabel, name, ylim=None, colors=None):
    xs = list(dict.fromkeys(df[x]))
    hs = list(dict.fromkeys(df[hue]))
    w = 0.8 / len(hs)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(xs) + 2), 4))
    for j, h in enumerate(hs):
        vals = [df[(df[x] == xv) & (df[hue] == h)][y].mean() for xv in xs]
        pos = np.arange(len(xs)) + (j - (len(hs) - 1) / 2) * w
        c = (colors or {}).get(h, SERIES[j % len(SERIES)])
        ax.bar(pos, vals, width=w * 0.92, color=c, label=h, edgecolor=SURFACE, linewidth=1)
        for p, v in zip(pos, vals):
            ax.text(p, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(np.arange(len(xs)), xs)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=min(len(hs), 4), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return save(fig, name)
