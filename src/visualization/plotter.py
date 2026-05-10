"""
src/visualization/plotter.py
Chức năng: Vẽ biểu đồ volatility và correlation heatmap.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import LABELS, COLORS, ROLLING_WINDOW, PATH_PROCESSED, PATH_FIGURES

BG      = "#0F1117"
PANEL   = "#1A1D2E"
BORDER  = "#333355"


def plot_volatility(vol: pd.DataFrame | None = None):
    """Vẽ biểu đồ chuỗi volatility theo thời gian."""
    if vol is None:
        vol = pd.read_csv(
            os.path.join(PATH_PROCESSED, "volatility_21d.csv"),
            index_col=0, parse_dates=True
        )
    os.makedirs(PATH_FIGURES, exist_ok=True)
    cols = list(vol.columns)

    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    for i, col in enumerate(cols[:5]):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        c  = COLORS.get(col, "#AAAAAA")
        ax.plot(vol.index, vol[col], color=c, linewidth=0.8)
        ax.fill_between(vol.index, vol[col], alpha=0.12, color=c)
        ax.axvline(pd.Timestamp("2020-03-15"), color="white",
                   ls="--", lw=0.7, alpha=0.6)
        ax.axvline(pd.Timestamp("2022-06-01"), color="#FF5252",
                   ls="--", lw=0.7, alpha=0.6)
        ax.set_facecolor(PANEL)
        ax.set_title(LABELS.get(col, col), color="white",
                     fontsize=11, fontweight="bold", pad=6)
        ax.tick_params(colors="#888888", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

    # Panel tổng hợp
    ax6 = fig.add_subplot(gs[2, :])
    ax6.set_facecolor(PANEL)
    for col in cols:
        v = vol[col]
        v_norm = (v - v.mean()) / v.std()
        ax6.plot(vol.index, v_norm, color=COLORS.get(col, "#AAAAAA"),
                 linewidth=0.9, label=LABELS.get(col, col), alpha=0.85)
    ax6.axvline(pd.Timestamp("2020-03-15"), color="white",
                ls="--", lw=0.8, alpha=0.5, label="COVID-19")
    ax6.axvline(pd.Timestamp("2022-06-01"), color="#FF5252",
                ls="--", lw=0.8, alpha=0.5, label="Crypto crash")
    ax6.legend(loc="upper left", fontsize=8, framealpha=0.3,
               labelcolor="white", facecolor="#222244")
    ax6.set_title("So sánh Volatility chuẩn hóa — Tất cả thị trường",
                  color="white", fontsize=11, fontweight="bold")
    ax6.tick_params(colors="#888888", labelsize=7)
    for sp in ax6.spines.values():
        sp.set_edgecolor(BORDER)

    fig.suptitle(
        f"Rolling Volatility {ROLLING_WINDOW} ngày (2018–2024)",
        color="white", fontsize=13, fontweight="bold", y=0.99
    )
    out = os.path.join(PATH_FIGURES, "volatility_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   ✅ Đã lưu: {out}")


def plot_correlation(vol: pd.DataFrame | None = None):
    """Vẽ correlation heatmap của các chuỗi volatility."""
    if vol is None:
        vol = pd.read_csv(
            os.path.join(PATH_PROCESSED, "volatility_21d.csv"),
            index_col=0, parse_dates=True
        )
    os.makedirs(PATH_FIGURES, exist_ok=True)

    corr = vol.corr()
    corr.index   = [LABELS.get(c, c) for c in corr.index]
    corr.columns = [LABELS.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=PANEL)
    ax.set_facecolor(PANEL)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor=BORDER,
                ax=ax, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 10, "color": "white"})
    ax.set_title("Correlation Matrix — Volatility 21 ngày",
                 color="white", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors="white", labelsize=9)
    plt.tight_layout()

    out = os.path.join(PATH_FIGURES, "correlation_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PANEL)
    plt.close()
    print(f"   ✅ Đã lưu: {out}")


if __name__ == "__main__":
    print("\n🎨 Đang vẽ biểu đồ...")
    plot_volatility()
    plot_correlation()
