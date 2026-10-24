import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

# ── Đường dẫn ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
PROC_DIR   = BASE_DIR / "data" / "processed"
TABLE_DIR  = BASE_DIR / "results" / "tables"
FIG_DIR    = BASE_DIR / "results" / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Mapping tên cột → nhãn hiển thị ───────────────────────────────────────
RETURN_COLS = [
    "return_vnindex",
    "return_btc",
    "return_oil",
    "return_gold",
    "return_sp500",
    "return_sse",
    "return_vix",
]
LABELS = {
    "return_vnindex" : "VNI",
    "return_btc"     : "BTC",
    "return_oil"     : "OIL",
    "return_gold"    : "GOLD",
    "return_sp500"   : "SP500",
    "return_sse"     : "SSE",
    "return_vix"     : "VIX",
}

# Màu sắc cho từng asset
COLORS = {
    "return_vnindex" : "#E63946",
    "return_btc"     : "#F4A261",
    "return_oil"     : "#2A9D8F",
    "return_gold"    : "#E9C46A",
    "return_sp500"   : "#457B9D",
    "return_sse"     : "#C77DFF",
    "return_vix"     : "#A8DADC",
}


# ═══════════════════════════════════════════════════════════════════════════
def compute_descriptive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính thống kê mô tả theo plan.md §4.1.
    Skewness và Kurtosis theo pandas (excess kurtosis, Fisher definition).
    """
    records = []
    for col in RETURN_COLS:
        s = df[col].dropna()
        jb_stat, jb_p = stats.jarque_bera(s)
        records.append({
            "Variable"   : LABELS[col],
            "N"          : len(s),
            "Mean (%)"   : round(s.mean(),       4),
            "Std (%)"    : round(s.std(ddof=1),  4),
            "Min (%)"    : round(s.min(),         4),
            "Max (%)"    : round(s.max(),         4),
            "Skewness"   : round(s.skew(),        4),
            "Kurtosis"   : round(s.kurtosis(),    4),   # excess kurtosis
            "JB Stat"    : round(jb_stat,         4),
            "JB p-value" : round(jb_p,            4),
            "Normal?"    : "No" if jb_p < 0.05 else "Yes",
        })
    return pd.DataFrame(records).set_index("Variable")


def plot_timeseries(df: pd.DataFrame):
    """Vẽ 7 chuỗi log-return theo thời gian (subplots)."""
    fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
    fig.suptitle("Log-Return (×100) của 7 Tài Sản — 2015–2026",
                 fontsize=15, fontweight="bold", y=1.01)

    for ax, col in zip(axes, RETURN_COLS):
        s = df.set_index("date")[col].dropna()
        ax.plot(s.index, s.values, lw=0.7, color=COLORS[col], alpha=0.85)
        ax.axhline(0, color="black", lw=0.5, ls="--")
        ax.set_ylabel(LABELS[col], fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Highlight 3 event windows
        events = [
            ("2020-01-01", "2021-12-31", "#FFB703", "COVID"),
            ("2022-03-01", "2022-12-31", "#FB8500", "Fed Hike"),
            ("2015-06-01", "2015-09-30", "#8ECAE6", "SSE Bubble"),
        ]
        for start, end, color, label in events:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=0.12, color=color, label=label if ax == axes[0] else "")

    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(loc="upper right", fontsize=8, framealpha=0.7)

    axes[-1].set_xlabel("Date", fontsize=10)
    plt.tight_layout()
    out = FIG_DIR / "return_timeseries.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_histogram(df: pd.DataFrame):
    """Vẽ histogram + KDE cho 7 biến (2 hàng × 4 cột)."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Phân phối Log-Return — Histogram & KDE",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(7)]

    for ax, col in zip(axes, RETURN_COLS):
        s = df[col].dropna()
        n_bins = min(80, max(30, len(s) // 50))
        ax.hist(s, bins=n_bins, density=True, color=COLORS[col],
                alpha=0.65, edgecolor="white", linewidth=0.3)

        # KDE overlay
        kde_x = np.linspace(s.min(), s.max(), 300)
        kde   = stats.gaussian_kde(s)
        ax.plot(kde_x, kde(kde_x), color="black", lw=1.5)

        # Normal distribution overlay (for comparison)
        norm_x  = np.linspace(s.min(), s.max(), 300)
        norm_y  = stats.norm.pdf(norm_x, s.mean(), s.std())
        ax.plot(norm_x, norm_y, color="red", lw=1.2, ls="--", alpha=0.7)

        ax.set_title(LABELS[col], fontsize=11, fontweight="bold")
        ax.set_xlabel("Return (%)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    # Ẩn subplot thừa (slot thứ 8)
    if len(axes) < 8:
        try:
            fig.add_subplot(gs[1, 3]).axis("off")
        except Exception:
            pass

    # Legend chung
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", lw=1.5, label="KDE"),
        Line2D([0], [0], color="red",   lw=1.2, ls="--", label="Normal"),
    ]
    fig.legend(handles=legend_elements, loc="lower right",
               fontsize=9, framealpha=0.8)

    out = FIG_DIR / "return_histogram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'='*65}")
    print("  BƯỚC 04 — THỐNG KÊ MÔ TẢ")
    print(f"{'='*65}\n")

    # Đọc data
    in_path = PROC_DIR / "master_return.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {in_path}\n"
            "→ Hãy chạy 03_merge_process.py trước!"
        )
    df = pd.read_csv(in_path, parse_dates=["date"])
    print(f"  ✓ Đọc master_return.csv: {len(df)} quan sát  |  "
          f"{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}\n")

    # ── Thống kê mô tả ────────────────────────────────────────────────────
    print("▶ 1. Tính thống kê mô tả...")
    desc = compute_descriptive(df)
    out_csv = TABLE_DIR / "descriptive_stats.csv"
    desc.to_csv(out_csv, encoding="utf-8-sig")
    print(f"  → {out_csv}\n")
    print(desc.to_string())

    # ── Vẽ biểu đồ ────────────────────────────────────────────────────────
    print(f"\n▶ 2. Vẽ biểu đồ time-series...")
    plot_timeseries(df)

    print(f"\n▶ 3. Vẽ histogram + KDE...")
    plot_histogram(df)

    print(f"\n{'='*65}")
    print("✅ Bước 04 hoàn tất!")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
