"""Figures for the README and the command line."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import (  # noqa: E402
    FLUX_LABELS,
    atp_contributions,
    fixed_points,
    nullcline_grid,
)
from .model import default_params, simulate  # noqa: E402

COLORS = {"A": "#1f6fb2", "H": "#c0392b", "Rmt": "#7d5ba6", "Rnox": "#d4a017",
          "ATP": "#2c3e50"}


def fig_dynamics(res, path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(res["t"], res["A"], color=COLORS["A"], label="AMPK (A)")
    ax[0].plot(res["t"], res["H"], color=COLORS["H"], label="HIF-1 (H)")
    ax[0].set_title("Signaling")
    ax[1].plot(res["t"], res["Rmt"], color=COLORS["Rmt"], label="mitochondrial ROS")
    ax[1].plot(res["t"], res["Rnox"], color=COLORS["Rnox"], label="NOX ROS")
    ax[1].set_title("Reactive oxygen species")
    ax[2].plot(res["t"], res["ATP"], color=COLORS["ATP"], label="net ATP")
    ax[2].set_title("Energy balance")
    for a in ax:
        a.set_xlabel("time (model units)")
        a.legend(frameon=False, fontsize=8)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("level (model units)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_nullclines(p, res, path, a_max=750.0, n=80):
    a_vec, h_vec, dA, dH = nullcline_grid(p, a_max=a_max, h_max=a_max, n=n)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.contour(a_vec, h_vec, dA, levels=[0], colors=COLORS["A"], linewidths=2)
    ax.contour(a_vec, h_vec, dH, levels=[0], colors=COLORS["H"], linewidths=2)
    ax.plot(res["A"], res["H"], color="#888888", lw=1, label="trajectory from default start")
    fps = fixed_points(p, a_max=a_max, h_max=a_max)
    st = [f for f in fps if f["stability"] == "stable"]
    sd = [f for f in fps if f["stability"] != "stable"]
    ax.plot([f["A"] for f in st], [f["H"] for f in st], "o", color="black", ms=8,
            label="stable steady state", zorder=5)
    ax.plot([f["A"] for f in sd], [f["H"] for f in sd], "o", mfc="white", mec="black", ms=8,
            label="saddle", zorder=5)
    ax.plot([], [], color=COLORS["A"], label="dA/dt = 0")
    ax.plot([], [], color=COLORS["H"], label="dH/dt = 0")
    ax.set_xlabel("AMPK (A)")
    ax.set_ylabel("HIF-1 (H)")
    ax.set_title("AMPK and HIF-1 phase plane")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_atp_budget(res, p, path):
    contrib = atp_contributions(res, p)
    final = {k: float(v[-1]) for k, v in contrib.items()}
    order = sorted(final, key=final.get)
    vals = np.array([final[k] for k in order])
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh([f"{k}  {FLUX_LABELS[k]}" for k in order], vals,
            color=np.where(vals >= 0, "#2e8b57", "#b03a2e"))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("ATP at steady state (model units)")
    ax.set_title(f"ATP budget: production minus consumption = {vals.sum():.0f}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(out_dir="docs/img", t_end=200.0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = default_params()
    res = simulate(p, t_span=(0, t_end))
    fig_dynamics(res, out / "dynamics.png")
    fig_nullclines(p, res, out / "phase_plane.png")
    fig_atp_budget(res, p, out / "atp_budget.png")
    return out
