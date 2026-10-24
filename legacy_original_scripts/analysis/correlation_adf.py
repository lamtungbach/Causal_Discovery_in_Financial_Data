import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import adfuller
from pathlib import Path

# ── Đường dẫn ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
TABLE_DIR = BASE_DIR / "results" / "tables"
FIG_DIR   = BASE_DIR / "results" / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Cấu hình ───────────────────────────────────────────────────────────────
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

# Kỳ vọng dấu tương quan (theo plan §4.2) — dùng để in chú thích
EXPECTED_SIGN = {
    ("VNI", "OIL")  : "+",
    ("VNI", "GOLD") : "−",
    ("VNI", "BTC")  : "~0",
    ("VNI", "SP500"): "+",
    ("VNI", "SSE")  : "+",
    ("VNI", "VIX")  : "−",
}


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN 1 — TƯƠNG QUAN PEARSON
# ═══════════════════════════════════════════════════════════════════════════

def compute_correlation(df: pd.DataFrame):
    """
    Tính ma trận tương quan Pearson và ma trận p-value pairwise.
    Sử dụng pairwise complete observations (dropna cho từng cặp).
    """
    cols  = RETURN_COLS
    n     = len(cols)
    corr  = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)
    pval  = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                corr.loc[c1, c2] = 1.0
                pval.loc[c1, c2] = 0.0
                continue
            # Pairwise complete
            valid = df[[c1, c2]].dropna()
            r, p  = pearsonr(valid[c1], valid[c2])
            corr.loc[c1, c2] = round(r, 4)
            pval.loc[c1, c2] = round(p, 4)

    # Đổi tên index/columns → nhãn ngắn
    rename = {c: LABELS[c] for c in cols}
    corr   = corr.rename(index=rename, columns=rename)
    pval   = pval.rename(index=rename, columns=rename)
    return corr, pval


def plot_correlation_heatmap(corr: pd.DataFrame, pval: pd.DataFrame):
    """
    Vẽ heatmap tương quan Pearson.
    - Ô có p < 0.05: viền đậm
    - Ô có p < 0.01: dấu *** trong annotation
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    # Tạo annotation: hệ số + mức ý nghĩa
    annot = pd.DataFrame("", index=corr.index, columns=corr.columns)
    for i in corr.index:
        for j in corr.columns:
            r = corr.loc[i, j]
            p = pval.loc[i, j]
            if i == j:
                annot.loc[i, j] = "1.00"
            else:
                stars = ""
                if p < 0.01:
                    stars = "***"
                elif p < 0.05:
                    stars = "**"
                elif p < 0.10:
                    stars = "*"
                annot.loc[i, j] = f"{r:.3f}{stars}"

    mask = np.eye(len(corr), dtype=bool)   # che đường chéo chính
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr.astype(float),
        ax=ax,
        annot=annot,
        fmt="",
        cmap=cmap,
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.6,
        linecolor="#cccccc",
        annot_kws={"size": 9},
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )

    ax.set_title(
        "Ma trận Tương quan Pearson — Log-Return (2015–2026)\n"
        "*** p<0.01  ** p<0.05  * p<0.10",
        fontsize=12, fontweight="bold", pad=14,
    )
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

    plt.tight_layout()
    out = FIG_DIR / "correlation_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN 2 — KIỂM ĐỊNH ADF
# ═══════════════════════════════════════════════════════════════════════════

def run_adf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chạy ADF test với lag tự động (AIC) cho 7 chuỗi log-return.
    Kỳ vọng tất cả đều dừng I(0) vì đây là return (không phải giá).
    """
    records = []
    for col in RETURN_COLS:
        s      = df[col].dropna()
        result = adfuller(s, autolag="AIC")
        adf_stat, p_val, n_lags, n_obs = result[0], result[1], result[2], result[3]
        cv     = result[4]   # critical values dict

        records.append({
            "Variable"       : LABELS[col],
            "ADF Statistic"  : round(adf_stat,    4),
            "p-value"        : round(p_val,        4),
            "Lags Used"      : n_lags,
            "N Obs"          : n_obs,
            "CV 1%"          : round(cv["1%"],     4),
            "CV 5%"          : round(cv["5%"],     4),
            "CV 10%"         : round(cv["10%"],    4),
            "Stationary 5%?" : "✓ I(0)" if p_val < 0.05 else "✗ Non-stationary",
        })

    return pd.DataFrame(records).set_index("Variable")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*65}")
    print("  BƯỚC 05 — TƯƠNG QUAN PEARSON & KIỂM ĐỊNH ADF")
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

    # ─── PHẦN 1: Tương quan ───────────────────────────────────────────────
    print("▶ 1. Tính ma trận tương quan Pearson (pairwise complete obs)...")
    corr, pval = compute_correlation(df)

    corr.to_csv(TABLE_DIR / "correlation_matrix.csv", encoding="utf-8-sig")
    pval.to_csv(TABLE_DIR / "correlation_pvalue.csv", encoding="utf-8-sig")
    print(f"  → {TABLE_DIR / 'correlation_matrix.csv'}")
    print(f"  → {TABLE_DIR / 'correlation_pvalue.csv'}\n")

    # In bảng tương quan với VNI để đối chiếu kỳ vọng
    print("  ── Tương quan với VNI ──")
    vni_corr = corr["VNI"].drop("VNI")
    vni_pval = pval["VNI"].drop("VNI")
    for asset in vni_corr.index:
        r   = vni_corr[asset]
        p   = vni_pval[asset]
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "ns"))
        exp = EXPECTED_SIGN.get(("VNI", asset), "?")
        match = "✓" if (
            (exp == "+" and r > 0) or
            (exp == "−" and r < 0) or
            (exp == "~0" and abs(r) < 0.1)
        ) else "✗"
        print(f"    VNI ↔ {asset:<6}  r={r:+.4f}  p={p:.4f} {sig:<3}  "
              f"kỳ vọng={exp}  {match}")

    print(f"\n▶ 2. Vẽ heatmap tương quan...")
    plot_correlation_heatmap(corr, pval)

    # ─── PHẦN 2: ADF Test ─────────────────────────────────────────────────
    print(f"\n▶ 3. Kiểm định tính dừng ADF (Augmented Dickey-Fuller)...")
    adf_df = run_adf(df)
    out_adf = TABLE_DIR / "adf_test.csv"
    adf_df.to_csv(out_adf, encoding="utf-8-sig")
    print(f"  → {out_adf}\n")
    print(adf_df[["ADF Statistic", "p-value", "Lags Used", "CV 5%", "Stationary 5%?"]].to_string())

    # Kiểm tra kỳ vọng: tất cả phải dừng
    non_stationary = adf_df[adf_df["Stationary 5%?"].str.startswith("✗")]
    if non_stationary.empty:
        print("\n  ✅ Tất cả 7 chuỗi log-return đều dừng I(0) tại mức 5% — đúng kỳ vọng!")
    else:
        print(f"\n  ⚠ {len(non_stationary)} chuỗi KHÔNG dừng tại 5%: "
              f"{list(non_stationary.index)}")
        print("     → Kiểm tra lại dữ liệu hoặc thử dùng KPSS test bổ sung.")

    print(f"\n{'='*65}")
    print("✅ Bước 05 hoàn tất!")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
