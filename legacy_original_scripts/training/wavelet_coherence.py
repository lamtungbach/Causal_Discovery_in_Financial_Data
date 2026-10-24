import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from pathlib import Path

try:
    import pycwt as wavelet
except ImportError:
    raise ImportError(
        "Thiếu thư viện pycwt!\n"
        "Cài đặt: pip install pycwt"
    )

# ──────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parent.parent.parent

PROC_DIR  = BASE_DIR / "data" / "processed"
TABLE_DIR = BASE_DIR / "results" / "tables"
FIG_DIR   = BASE_DIR / "results" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

PAIRS = {
    "btc_vni"  : ("return_btc",     "return_vnindex"),
    "oil_vni"  : ("return_oil",     "return_vnindex"),
    "gold_vni" : ("return_gold",    "return_vnindex"),
    "sp500_vni": ("return_sp500",   "return_vnindex"),
    "sse_vni"  : ("return_sse",     "return_vnindex"),
    "vix_vni"  : ("return_vix",     "return_vnindex"),
}

PAIR_TITLES = {
    "btc_vni"  : "Bitcoin (BTC) & VN-Index (VNI)",
    "oil_vni"  : "Dầu WTI (OIL) & VN-Index (VNI)",
    "gold_vni" : "Vàng (GOLD) & VN-Index (VNI)",
    "sp500_vni": "S&P 500 (SP500) & VN-Index (VNI)",
    "sse_vni"  : "Shanghai Composite (SSE) & VN-Index (VNI)",
    "vix_vni"  : "CBOE VIX & VN-Index (VNI)",
}

PERIODS = {
    "pre_covid" : ("2015-01-01", "2019-12-31", "Trước COVID-19 (2015–2019)"),
    "covid"     : ("2020-01-01", "2021-12-31", "COVID-19 (2020–2021)"),
    "post_covid": ("2022-01-01", "2026-05-12", "Hậu COVID & Nga–Ukraine (2022–2026)"),
}

# Frequency bands (đơn vị: ngày giao dịch)
FREQ_BANDS = {
    "short"  : (4,   16),
    "medium" : (16,  64),
    "long"   : (64,  256),
}

# Wavelet parameters
DT  = 1
DJ  = 1 / 12
S0  = 2 * DT

# ~7 octaves -> max scale ≈ 256 ngày
J = int(7 / DJ)

SIG_LEVEL = 0.95


# ══════════════════════════════════════════════════════════════════════
# COMPUTE WTC
# ══════════════════════════════════════════════════════════════════════

def compute_wtc(x: np.ndarray, y: np.ndarray):

    # Standardize
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)

    mother = wavelet.Morlet(6)

    WCT, aWCT, coi, freqs, sig = wavelet.wct(
        x,
        y,
        dt=DT,
        dj=DJ,
        s0=S0,
        J=J,
        significance_level=SIG_LEVEL,
        wavelet=mother,
        normalize=False,
    )

    return WCT, aWCT, coi, freqs, sig


# ══════════════════════════════════════════════════════════════════════
# PLOT WTC
# ══════════════════════════════════════════════════════════════════════

def plot_wtc(
    WCT,
    aWCT,
    coi,
    freqs,
    sig,
    time_index,
    title,
    save_path,
):

    periods = 1.0 / freqs
    t = np.arange(len(time_index))

    fig, ax = plt.subplots(figsize=(13, 6))

    # ──────────────────────────────────────────────────────────────
    # MAIN HEATMAP
    # ──────────────────────────────────────────────────────────────

    cmap = plt.cm.RdYlGn
    levels = np.linspace(0, 1, 101)

    cf = ax.contourf(
        t,
        np.log2(periods),
        WCT,
        levels=levels,
        cmap=cmap,
        extend="both",
    )

    # ──────────────────────────────────────────────────────────────
    # SIGNIFICANCE CONTOUR
    # ──────────────────────────────────────────────────────────────

    sig95 = np.ones_like(WCT)

    for j in range(len(periods)):
        sig95[j, :] = WCT[j, :] / sig[j]

    ax.contour(
        t,
        np.log2(periods),
        sig95,
        levels=[1],
        colors="black",
        linewidths=1.5,
    )

    # ──────────────────────────────────────────────────────────────
    # COI
    # ──────────────────────────────────────────────────────────────

    coi = np.maximum(coi, periods[0])
    coi_log2 = np.log2(coi)

    ax.fill_between(
        t,
        coi_log2,
        np.log2(periods[-1]),
        color="white",
        alpha=0.55,
        hatch="//",
    )

    ax.plot(
        t,
        coi_log2,
        "k--",
        lw=0.8,
        alpha=0.7,
    )

    # ──────────────────────────────────────────────────────────────
    # PHASE ARROWS
    # ──────────────────────────────────────────────────────────────

    nt = len(t)
    ns = len(periods)

    step_t = max(1, nt // 35)
    step_s = max(1, ns // 14)

    for i in range(0, nt, step_t):
        for j in range(0, ns, step_s):

            if WCT[j, i] < 0.5:
                continue

            angle = aWCT[j, i]

            dx = np.cos(angle) * 0.35
            dy = np.sin(angle) * 0.35

            ax.annotate(
                "",
                xy=(
                    t[i] + dx,
                    np.log2(periods[j]) + dy
                ),
                xytext=(
                    t[i],
                    np.log2(periods[j])
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    lw=0.8,
                ),
                annotation_clip=True,
            )

    # ──────────────────────────────────────────────────────────────
    # FREQUENCY BAND LINES
    # ──────────────────────────────────────────────────────────────

    for _, (lo, hi) in FREQ_BANDS.items():

        for bound in [lo, hi]:

            if periods[0] <= bound <= periods[-1]:

                ax.axhline(
                    np.log2(bound),
                    color="gray",
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.7,
                )

    # ──────────────────────────────────────────────────────────────
    # X AXIS
    # ──────────────────────────────────────────────────────────────

    n_years = (
        (time_index.iloc[-1] - time_index.iloc[0]).days // 365
    ) + 1

    step_yr = max(1, n_years // 6)

    tick_vals = []
    tick_labs = []

    for idx, dt in enumerate(time_index):

        if (
            dt.month == 1
            and dt.day <= 7
            and dt.year % step_yr == 0
        ):
            tick_vals.append(idx)
            tick_labs.append(str(dt.year))

    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labs, fontsize=9)

    ax.set_xlim(0, len(t) - 1)

    # ──────────────────────────────────────────────────────────────
    # Y AXIS
    # ──────────────────────────────────────────────────────────────

    ytick_scales = [4, 8, 16, 32, 64, 128, 256]

    ytick_vals = [
        np.log2(s)
        for s in ytick_scales
        if periods[0] <= s <= max(periods)
    ]

    ytick_labs = [
        str(s)
        for s in ytick_scales
        if periods[0] <= s <= max(periods)
    ]

    ax.set_yticks(ytick_vals)
    ax.set_yticklabels(ytick_labs, fontsize=9)

    ax.set_ylim(
        np.log2(periods[0]),
        np.log2(periods[-1]),
    )

    ax.set_xlabel("Thời gian", fontsize=10)
    ax.set_ylabel("Thang thời gian (ngày, log₂)", fontsize=10)

    # ──────────────────────────────────────────────────────────────
    # COLORBAR
    # ──────────────────────────────────────────────────────────────

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.Normalize(0, 1)
    )

    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=ax,
        fraction=0.03,
        pad=0.02,
    )

    cbar.set_label(
        "R² (Wavelet Coherence)",
        fontsize=9,
    )

    cbar.ax.tick_params(labelsize=8)

    # ──────────────────────────────────────────────────────────────
    # LEGEND
    # ──────────────────────────────────────────────────────────────

    band_patches = [
        mpatches.Patch(
            color="none",
            label="── Frequency Bands ──"
        ),

        mpatches.Patch(
            color="#aaaaaa",
            alpha=0.4,
            label="Ngắn: 4–16 ngày"
        ),

        mpatches.Patch(
            color="#888888",
            alpha=0.4,
            label="Trung: 16–64 ngày"
        ),

        mpatches.Patch(
            color="#555555",
            alpha=0.4,
            label="Dài: 64–256 ngày"
        ),

        mpatches.Patch(
            color="white",
            hatch="//",
            alpha=0.7,
            label="COI"
        ),
    ]

    ax.legend(
        handles=band_patches,
        loc="lower right",
        fontsize=7.5,
        framealpha=0.7,
    )

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ══════════════════════════════════════════════════════════════════════
# AVERAGE COHERENCE
# ══════════════════════════════════════════════════════════════════════

def avg_coherence_by_band(
    WCT: np.ndarray,
    freqs: np.ndarray,
):

    periods = 1.0 / freqs

    result = {}

    for band_name, (lo, hi) in FREQ_BANDS.items():

        mask = (
            (periods >= lo)
            & (periods < hi)
        )

        if mask.sum() == 0:
            result[band_name] = np.nan
        else:
            result[band_name] = float(
                np.nanmean(WCT[mask, :])
            )

    result["overall"] = float(np.nanmean(WCT))

    return result


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():

    print(f"\n{'='*65}")
    print("  BƯỚC 06 — WAVELET COHERENCE")
    print(f"{'='*65}\n")

    in_path = PROC_DIR / "master_return.csv"

    if not in_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {in_path}\n"
            "→ Hãy chạy 03_merge_process.py trước!"
        )

    df = pd.read_csv(
        in_path,
        parse_dates=["date"]
    )

    print(f"✓ Đọc dữ liệu: {len(df)} quan sát\n")

    avg_coherence_records = []

    total_plots = len(PAIRS) * len(PERIODS)
    done = 0

    for pair_name, (col_x, col_y) in PAIRS.items():

        for period_key, (
            start,
            end,
            period_label
        ) in PERIODS.items():

            done += 1

            print(
                f"[{done:>2}/{total_plots}] "
                f"{pair_name.upper()} | {period_label}",
                end=" ",
                flush=True,
            )

            mask = (
                (df["date"] >= start)
                & (df["date"] <= end)
            )

            df_p = (
                df[mask]
                .copy()
                .reset_index(drop=True)
            )

            if len(df_p) < 60:
                print("⚠ Không đủ dữ liệu")
                continue

            x = (
                df_p[col_x]
                .ffill()
                .fillna(0)
                .values
            )

            y = (
                df_p[col_y]
                .ffill()
                .fillna(0)
                .values
            )

            try:

                WCT, aWCT, coi, freqs, sig = compute_wtc(x, y)

            except Exception as e:

                print(f"✗ Lỗi: {e}")
                continue

            title = (
                f"Wavelet Coherence: "
                f"{PAIR_TITLES[pair_name]}\n"
                f"{period_label}"
            )

            save_path = (
                FIG_DIR
                / f"wavelet_{pair_name}_{period_key}.png"
            )

            plot_wtc(
                WCT,
                aWCT,
                coi,
                freqs,
                sig,
                df_p["date"],
                title,
                save_path,
            )

            avg_r2 = avg_coherence_by_band(
                WCT,
                freqs,
            )

            avg_coherence_records.append({
                "pair"           : pair_name.upper(),
                "period"         : period_key,
                "n_obs"          : len(df_p),
                "avg_R2_short"   : round(avg_r2["short"], 4),
                "avg_R2_medium"  : round(avg_r2["medium"], 4),
                "avg_R2_long"    : round(avg_r2["long"], 4),
                "avg_R2_overall" : round(avg_r2["overall"], 4),
            })

            print(
                f"✓ overall={avg_r2['overall']:.3f}"
            )

    # SAVE SUMMARY

    df_avg = pd.DataFrame(avg_coherence_records)

    out_avg = (
        TABLE_DIR
        / "avg_coherence_summary.csv"
    )

    df_avg.to_csv(
        out_avg,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n→ Saved: {out_avg}")

    # SUMMARY TABLE

    print(f"\n{'='*65}")
    print("TÓM TẮT R²")
    print(f"{'='*65}")

    pivot = (
        df_avg
        .pivot_table(
            index="pair",
            columns="period",
            values="avg_R2_overall",
            aggfunc="first",
        )
        .round(3)
    )

    print(pivot.to_string())

    print(f"\n{'='*65}")
    print("✅ Hoàn tất WTC")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()