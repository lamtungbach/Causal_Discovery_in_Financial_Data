"""
src/data/validator.py
=====================
Kiểm định tính dừng (ADF, KPSS) và thống kê mô tả.

Fix so với phiên bản cũ:
  ✅ Dùng absolute path (pathlib) — nhất quán với preprocessor.py
  ✅ Tự động tạo volatility_stationary.csv cho pipeline
  ✅ LABELS key khớp với VNINDEX (sau khi sửa config)
  ✅ Tương thích với NeuralGrangerCausality pipeline
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[2]   # D:\KLTN
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR   = BASE_DIR / "data" / "results"

sys.path.insert(0, str(BASE_DIR))
from config import LABELS

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Thống kê mô tả
# ─────────────────────────────────────────────────────────────────────────────

def summary_stats(vol: pd.DataFrame) -> pd.DataFrame:
    """
    Thống kê mô tả đầy đủ cho Bảng 4.1 (Chapter 4).
    Columns: N, Mean, Std, Min, Max, Skewness, Kurtosis, JB Stat, JB p-value
    """
    rows = []
    for col in vol.columns:
        s = vol[col].dropna()
        jb_stat, jb_p = stats.jarque_bera(s)
        rows.append({
            "Thị trường" : LABELS.get(col, col),
            "N"          : len(s),
            "Mean"       : round(s.mean(), 6),
            "Std"        : round(s.std(),  6),
            "Min"        : round(s.min(),  6),
            "Max"        : round(s.max(),  6),
            "Skewness"   : round(stats.skew(s),     4),
            "Kurtosis"   : round(stats.kurtosis(s), 4),
            "JB Stat"    : round(jb_stat, 2),
            "JB p-value" : round(jb_p,    4),
        })
    return pd.DataFrame(rows).set_index("Thị trường")


# ─────────────────────────────────────────────────────────────────────────────
# Kiểm định tính dừng
# ─────────────────────────────────────────────────────────────────────────────

def test_stationarity(vol: pd.DataFrame) -> pd.DataFrame:
    """
    Kiểm định ADF và KPSS cho từng chuỗi volatility.

    Lý thuyết:
        ADF : H0 = phi dừng → p < 0.05 → bác bỏ H0 → dừng ✅
        KPSS: H0 = dừng     → p > 0.05 → không bác bỏ → dừng ✅

    Kết luận:
        Cả hai dừng          → "Dừng"
        ADF dừng, KPSS không → "Có thể dừng"  (phổ biến với rolling vol)
        ADF không dừng       → "Phi dừng"
    """
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError:
        raise ImportError("Chạy: pip install statsmodels")

    sep = "─" * 95
    print(f"\n   {sep}")
    print(f"   {'Thị trường':<18} {'ADF p':>8} {'ADF':>14} {'KPSS p':>8} {'KPSS':>14} {'Kết luận':>14}")
    print(f"   {sep}")

    rows = []
    for col in vol.columns:
        s     = vol[col].dropna()
        label = LABELS.get(col, col)

        # ── ADF ───────────────────────────────────────────────────────────────
        adf_stat, adf_p, _, _, _, _ = adfuller(s, autolag="AIC")
        adf_ok  = adf_p < 0.05
        adf_lbl = "✅ Dừng" if adf_ok else "❌ Phi dừng"

        # ── KPSS ──────────────────────────────────────────────────────────────
        try:
            kpss_stat, kpss_p, _, _ = kpss(s, regression="c", nlags="auto")
            kpss_ok  = kpss_p >= 0.05
            kpss_lbl = "✅ Dừng" if kpss_ok else "⚠️  Phi dừng"
            kpss_p_s = f"{kpss_p:.4f}"
        except Exception:
            kpss_stat = kpss_p = np.nan
            kpss_ok   = None
            kpss_lbl  = "N/A"
            kpss_p_s  = "N/A"

        # ── Kết luận tổng hợp ─────────────────────────────────────────────────
        if adf_ok and (kpss_ok is None or kpss_ok):
            verdict     = "✅ Dừng"
            verdict_csv = "Dừng"
        elif adf_ok and not kpss_ok:
            verdict     = "⚠️  Có thể dừng"
            verdict_csv = "Có thể dừng"
        else:
            verdict     = "❌ Phi dừng"
            verdict_csv = "Phi dừng"

        print(f"   {label:<18} {adf_p:>8.4f} {adf_lbl:>14} "
              f"{kpss_p_s:>8} {kpss_lbl:>14} {verdict:>14}")

        rows.append({
            "Thị trường"   : label,
            "ADF stat"     : round(adf_stat,  4),
            "ADF p-value"  : round(adf_p,     4),
            "ADF kết quả"  : "Dừng" if adf_ok else "Phi dừng",
            "KPSS stat"    : round(kpss_stat, 4) if not np.isnan(kpss_stat) else None,
            "KPSS p-value" : round(kpss_p,    4) if not np.isnan(kpss_p)    else None,
            "KPSS kết quả" : ("Dừng" if kpss_ok else "Phi dừng") if kpss_ok is not None else "N/A",
            "Kết luận"     : verdict_csv,
        })

    print(f"   {sep}")
    print(f"\n   ℹ️  Ghi chú:")
    print(f"   'Có thể dừng' = ADF dừng nhưng KPSS phi dừng.")
    print(f"   Đây là hiện tượng phổ biến với rolling volatility tài chính.")
    print(f"   Pipeline sử dụng volatility GỐC (không lấy sai phân).")

    return pd.DataFrame(rows).set_index("Thị trường")


# ─────────────────────────────────────────────────────────────────────────────
# Correlation matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_correlation(vol: pd.DataFrame) -> pd.DataFrame:
    """Ma trận tương quan Pearson — dùng cho Hình 4.3 (Correlation Heatmap)."""
    return vol.rename(columns=LABELS).corr()


# ─────────────────────────────────────────────────────────────────────────────
# Tạo volatility_stationary.csv cho pipeline
# ─────────────────────────────────────────────────────────────────────────────

def create_stationary_file(vol: pd.DataFrame, stat_df: pd.DataFrame) -> None:
    """
    Tạo volatility_stationary.csv để các model (NOTEARS, PC, Granger) sử dụng.

    Vì rolling volatility thường đã dừng theo lý thuyết tài chính,
    ta dùng trực tiếp volatility gốc (không lấy sai phân).
    Nếu có cột thực sự phi dừng (ADF p > 0.05), log cảnh báo để người dùng biết.
    """
    stationary_path = PROCESSED_DIR / "volatility_stationary.csv"

    # Kiểm tra cột phi dừng thực sự
    truly_nonstationary = stat_df[stat_df["Kết luận"] == "Phi dừng"].index.tolist()
    if truly_nonstationary:
        print(f"\n   ⚠️  Cảnh báo: Các cột phi dừng theo ADF: {truly_nonstationary}")
        print(f"   → Vẫn dùng volatility gốc. Ghi chú trong thesis phần 4.2.3.")
    else:
        print(f"\n   ✅ Tất cả chuỗi được coi là dừng — dùng volatility gốc.")

    vol.to_csv(stationary_path)
    print(f"   ✅ Đã tạo: {stationary_path} | shape: {vol.shape}")
    print(f"   → Đây là input cho NOTEARS, PC Algorithm, Neural Granger Causality")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(vol: pd.DataFrame = None) -> tuple:
    """
    Chạy toàn bộ kiểm định thống kê và tạo file output.

    Parameters
    ----------
    vol : pd.DataFrame  volatility đã tính (nếu None → tự load từ file)

    Returns
    -------
    desc    : pd.DataFrame  thống kê mô tả
    stat_df : pd.DataFrame  kết quả ADF + KPSS
    """
    print("\n🔬 Validator — Kiểm định tính dừng & Thống kê mô tả")
    print("=" * 55)

    # Tạo thư mục output
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load volatility nếu chưa có ───────────────────────────────────────────
    if vol is None:
        vol_path = PROCESSED_DIR / "volatility_21d.csv"
        if not vol_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {vol_path}\n"
                f"Chạy trước: python src/data/preprocessor.py"
            )
        vol = pd.read_csv(vol_path, index_col=0, parse_dates=True)
        print(f"\n   Load volatility từ: {vol_path}")
        print(f"   Shape: {vol.shape}")
        print(f"   Cột  : {list(vol.columns)}")

    # ── Thống kê mô tả ────────────────────────────────────────────────────────
    print("\n   📊 Thống kê mô tả (Volatility annualized):")
    desc = summary_stats(vol)
    print(desc.to_string())
    desc.to_csv(RESULTS_DIR / "summary_stats.csv")

    # ── Kiểm định tính dừng ───────────────────────────────────────────────────
    print("\n   📋 Kiểm định tính dừng (ADF & KPSS):")
    stat_df = test_stationarity(vol)
    stat_df.to_csv(RESULTS_DIR / "adf_kpss_results.csv")

    # ── Correlation matrix ────────────────────────────────────────────────────
    corr = compute_correlation(vol)
    corr.to_csv(RESULTS_DIR / "correlation_matrix.csv")

    # ── Tạo volatility_stationary.csv ─────────────────────────────────────────
    create_stationary_file(vol, stat_df)

    # ── Tóm tắt ───────────────────────────────────────────────────────────────
    n_stationary = stat_df["Kết luận"].isin(["Dừng", "Có thể dừng"]).sum()
    n_total      = len(stat_df)

    print(f"\n   ✅ Đã lưu:")
    print(f"      {RESULTS_DIR / 'summary_stats.csv'}")
    print(f"      {RESULTS_DIR / 'adf_kpss_results.csv'}")
    print(f"      {RESULTS_DIR / 'correlation_matrix.csv'}")
    print(f"      {PROCESSED_DIR / 'volatility_stationary.csv'}")
    print(f"\n   📈 Kết quả: {n_stationary}/{n_total} thị trường được coi là dừng")
    print(f"   → Pipeline sẵn sàng chạy Neural Granger Causality ✅")

    return desc, stat_df


if __name__ == "__main__":
    run_validation()