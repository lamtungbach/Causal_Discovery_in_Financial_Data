"""
src/visualization/time_series.py
Plot chuỗi thời gian volatility, correlation heatmap, threshold sensitivity.

Đặt file tại: src/visualization/time_series.py
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

_COLORS = {
    "SP500"  : "#E63946",
    "VNINDEX": "#F4A261",
    "Gold"   : "#FFD166",
    "WTI_Oil": "#06D6A0",
    "Bitcoin": "#118AB2",
}
_LABELS = {
    "SP500"  : "S&P 500",
    "VNINDEX": "VN-Index",
    "Gold"   : "Vàng (Gold)",
    "WTI_Oil": "Dầu WTI",
    "Bitcoin": "Bitcoin",
}

try:
    from config import MARKET_COLORS as _COLORS, MARKET_LABELS as _LABELS, FIG_DPI
except ImportError:
    FIG_DPI = 150

# Sự kiện lớn cần đánh dấu trên timeline
_EVENTS = {
    "COVID-19\n(3/2020)"       : "2020-03-01",
    "Crypto Crash\n(5/2022)"   : "2022-05-01",
    "Banking Crisis\n(3/2023)" : "2023-03-01",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rolling Volatility Time Series — Chương 4.1
# ─────────────────────────────────────────────────────────────────────────────

def plot_volatility_series(
    vol_df     : pd.DataFrame,
    output_path: Optional[str] = None,
    mark_events: bool = True,
    figsize    : tuple = (14, 12),
) -> "plt.Figure":
    """
    Vẽ chuỗi thời gian rolling volatility 21 ngày của 5 thị trường.

    Mỗi thị trường = 1 subplot riêng, dùng chung trục x.
    Các sự kiện lớn (COVID, crypto crash, banking crisis) được đánh dấu
    bằng đường kẻ dọc để dễ nhận diện spike volatility.

    Parameters
    ----------
    vol_df      : pd.DataFrame  shape (n_days, 5), index = DatetimeIndex
    output_path : str, optional
    mark_events : bool          có vẽ đường kẻ dọc sự kiện không
    figsize     : tuple

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> fig = plot_volatility_series(
    ...     vol_df      = preprocessor.volatility_,
    ...     output_path = "reports/figures/volatility_series.png",
    ... )
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        raise ImportError("pip install matplotlib")

    cols = list(vol_df.columns)
    fig, axes = plt.subplots(len(cols), 1, figsize=figsize, sharex=True)
    if len(cols) == 1:
        axes = [axes]

    fig.suptitle(
        "Rolling Volatility 21 ngày — 5 Thị trường Tài chính (2018–2024)",
        fontsize   = 14,
        fontweight = "bold",
    )

    for ax, col in zip(axes, cols):
        color = _COLORS.get(col, "#888888")
        label = _LABELS.get(col, col)

        ax.plot(vol_df.index, vol_df[col],
                color=color, linewidth=1.1, label=label)
        ax.fill_between(vol_df.index, vol_df[col],
                        alpha=0.13, color=color)

        # Highlight max (spike volatility)
        idx_max = vol_df[col].idxmax()
        ax.axvline(idx_max, color=color, linestyle="--",
                   linewidth=0.8, alpha=0.5)
        ax.annotate(
            f"Max\n{idx_max.strftime('%m/%Y')}",
            xy         = (idx_max, vol_df[col].max()),
            xytext     = (10, -15),
            textcoords = "offset points",
            fontsize   = 7,
            color      = color,
            alpha      = 0.8,
        )

        ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=60, va="center")
        ax.grid(True, alpha=0.22, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if mark_events:
            for evt_label, evt_date in _EVENTS.items():
                ax.axvline(pd.Timestamp(evt_date),
                           color="gray", linestyle=":", linewidth=1.0, alpha=0.65)

    # Ghi chú sự kiện chỉ trên subplot đầu tiên
    if mark_events:
        y_top = axes[0].get_ylim()[1]
        for evt_label, evt_date in _EVENTS.items():
            axes[0].text(
                pd.Timestamp(evt_date), y_top * 0.88,
                evt_label, fontsize=7, ha="center", color="dimgray",
            )

    # Format trục x
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].set_xlabel("Năm", fontsize=10)

    plt.tight_layout()
    _save_fig(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Correlation Heatmap — Chương 4.1
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    vol_df     : pd.DataFrame,
    output_path: Optional[str] = None,
    figsize    : tuple = (8, 7),
) -> "plt.Figure":
    """
    Heatmap tương quan Pearson giữa 5 chuỗi volatility.

    Dùng làm bước khởi đầu (Chương 4.1) trước khi đi vào causal discovery —
    để thấy rằng tương quan ≠ nhân quả.

    Parameters
    ----------
    vol_df      : pd.DataFrame  chuỗi volatility 5 thị trường
    output_path : str, optional
    figsize     : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("pip install matplotlib seaborn")

    corr   = vol_df.corr()
    labels = [_LABELS.get(c, c) for c in corr.columns]

    # Mask tam giác trên để tránh trùng lặp
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask        = mask,
        annot       = True,
        fmt         = ".3f",
        cmap        = "RdYlGn",
        xticklabels = labels,
        yticklabels = labels,
        vmin        = -1,
        vmax        = 1,
        center      = 0,
        linewidths  = 0.5,
        ax          = ax,
        annot_kws   = {"size": 11},
        square      = True,
    )
    ax.set_title(
        "Ma trận Tương quan Volatility\n5 Thị trường Tài chính (2018–2024)",
        fontsize   = 13,
        fontweight = "bold",
        pad        = 12,
    )
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()

    _save_fig(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Threshold Sensitivity Plot — Chương 3.5.3
# ─────────────────────────────────────────────────────────────────────────────

def plot_threshold_sensitivity(
    sensitivity : List[dict],
    output_path : Optional[str] = None,
    figsize     : tuple = (9, 5),
    highlight   : Optional[float] = 0.2,
) -> "plt.Figure":
    """
    Vẽ số cạnh và graph density theo từng ngưỡng threshold của NOTEARS.

    Dùng để chọn threshold tối ưu (Chương 3.5.3) — threshold mà:
    - Không quá thưa (mất thông tin)
    - Không quá dày  (nhiễu)

    Parameters
    ----------
    sensitivity : List[dict]
        Output từ notears_model.threshold_sensitivity().
        Mỗi dict gồm: {"threshold": float, "n_edges": int, "density": float}
    output_path : str, optional
    figsize     : tuple
    highlight   : float, optional
        Giá trị threshold muốn highlight (đường kẻ dọc đỏ).
        Mặc định 0.2 (threshold được chọn trong khóa luận).

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> sensitivity = notears_model.threshold_sensitivity()
    >>> fig = plot_threshold_sensitivity(
    ...     sensitivity  = sensitivity,
    ...     output_path  = "reports/figures/threshold_sensitivity.png",
    ...     highlight    = 0.2,
    ... )
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("pip install matplotlib")

    thresholds = [r["threshold"] for r in sensitivity]
    n_edges    = [r["n_edges"]   for r in sensitivity]
    densities  = [r["density"]   for r in sensitivity]

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # Số cạnh (trục trái)
    line1, = ax1.plot(
        thresholds, n_edges,
        "b-o", linewidth=2, markersize=7, label="Số cạnh (edges)",
    )
    ax1.fill_between(thresholds, n_edges, alpha=0.08, color="blue")

    # Graph density (trục phải)
    line2, = ax2.plot(
        thresholds, densities,
        "r--s", linewidth=1.5, markersize=6, label="Graph Density",
    )

    # Highlight threshold được chọn
    if highlight and highlight in thresholds:
        idx  = thresholds.index(highlight)
        ax1.axvline(highlight, color="green", linestyle=":", linewidth=1.8, alpha=0.8)
        ax1.annotate(
            f"Chọn\nthreshold={highlight}",
            xy         = (highlight, n_edges[idx]),
            xytext     = (10, 15),
            textcoords = "offset points",
            fontsize   = 8,
            color      = "green",
            arrowprops = dict(arrowstyle="->", color="green", lw=1.2),
        )

    ax1.set_xlabel("Threshold cắt cạnh", fontsize=12)
    ax1.set_ylabel("Số cạnh (Edges)", color="blue",  fontsize=11)
    ax2.set_ylabel("Graph Density",    color="red",   fontsize=11)
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="red")

    lines  = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=9)

    ax1.set_title(
        "NOTEARS — Phân tích độ nhạy Threshold\n"
        "(Chọn threshold tối ưu cân bằng giữa sparsity và completeness)",
        fontsize   = 12,
        fontweight = "bold",
    )
    ax1.grid(True, alpha=0.28, linestyle="--")
    plt.tight_layout()

    _save_fig(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Descriptive Stats Bar Chart — Chương 4.1
# ─────────────────────────────────────────────────────────────────────────────

def plot_descriptive_stats(
    desc_df    : "pd.DataFrame",
    metric     : str = "Std",
    output_path: Optional[str] = None,
    figsize    : tuple = (9, 5),
) -> "plt.Figure":
    """
    Bar chart một metric thống kê mô tả (Mean / Std / Skewness / Kurtosis)
    cho 5 thị trường.

    Parameters
    ----------
    desc_df     : pd.DataFrame  output từ DataPreprocessor.describe()
    metric      : str           tên cột muốn vẽ
    output_path : str, optional
    figsize     : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("pip install matplotlib")

    if metric not in desc_df.columns:
        raise ValueError(f"Metric '{metric}' không tồn tại. Chọn: {list(desc_df.columns)}")

    markets = list(desc_df.index)
    values  = desc_df[metric].values
    colors  = [_COLORS.get(m, "#aaa") for m in markets]
    labels  = [_LABELS.get(m, m)      for m in markets]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white", width=0.55)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(
        f"Thống kê mô tả — {metric}\n5 Thị trường Tài chính (2018–2024)",
        fontsize   = 12,
        fontweight = "bold",
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    _save_fig(fig, output_path)
    return fig


# ─── Internal helper ─────────────────────────────────────────────────────────

def _save_fig(fig: "plt.Figure", path: Optional[str]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        logger.info(f"Đã lưu: {path}")
