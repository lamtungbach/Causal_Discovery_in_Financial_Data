from __future__ import annotations
import numpy as np
import pandas as pd


def _fallback_multiscale(series: np.ndarray, level: int = 6):
    """Fallback không cần PyWavelets: multi-scale moving-average details.
    Không thay thế MODWT về mặt lý thuyết, nhưng giúp pipeline vẫn chạy khi chưa cài PyWavelets.
    Khi viết khóa luận, nên cài PyWavelets và dùng SWT/MODWT.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    details = {}
    prev = x.copy()
    for j in range(1, level + 1):
        w = 2 ** j
        smooth = pd.Series(prev).rolling(w, min_periods=1, center=True).mean().values
        details[f"D{j}"] = prev - smooth
        prev = smooth
    return details, prev


def safe_swt(series: np.ndarray, wavelet: str = "db4", level: int = 6):
    """SWT được dùng như xấp xỉ MODWT trong PyWavelets.
    Nếu môi trường chưa có PyWavelets, dùng fallback multi-scale moving-average để code vẫn chạy.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    max_level = min(level, int(np.floor(np.log2(len(x)))) if len(x) > 0 else 1)
    if max_level < 1:
        raise ValueError("Chuỗi quá ngắn để phân rã multi-scale")
    try:
        import pywt
        n_valid = (len(x) // (2 ** max_level)) * (2 ** max_level)
        x = x[:n_valid]
        coeffs = pywt.swt(x, wavelet=wavelet, level=max_level, norm=True)
        details = {f"D{j}": cD for j, (_, cD) in enumerate(reversed(coeffs), start=1)}
        smooth = coeffs[0][0]
        return details, smooth
    except ImportError:
        print("[WARN] PyWavelets chưa được cài. Đang dùng fallback moving-average decomposition.")
        return _fallback_multiscale(x, max_level)


def modwt_band_variance(series: np.ndarray, band_levels: dict, wavelet: str = "db4", level: int = 6):
    details, smooth = safe_swt(series, wavelet=wavelet, level=level)
    variances = {name: 0.0 for name in band_levels}
    for band, levels in band_levels.items():
        variances[band] = float(sum(np.var(details.get(f"D{j}", np.array([0.0])), ddof=1) for j in levels))
    variances["smooth"] = float(np.var(smooth, ddof=1))
    total = sum(v for v in variances.values() if np.isfinite(v))
    if total <= 0:
        return {k: 0.0 for k in variances}
    return {k: v / total * 100 for k, v in variances.items()}


def export_decomposed_components(df: pd.DataFrame, cols, band_levels, wavelet="db4", level=6):
    """Trả về DataFrame chứa các thành phần wavelet/multiscale band-level cho từng biến."""
    min_len = None
    temp = {}
    for col in cols:
        details, smooth = safe_swt(df[col].dropna().values, wavelet=wavelet, level=level)
        bands = {}
        for band, levels in band_levels.items():
            arrs = [details[f"D{j}"] for j in levels if f"D{j}" in details]
            bands[band] = np.sum(arrs, axis=0) if arrs else np.zeros_like(smooth)
        bands["smooth"] = smooth
        temp[col] = bands
        min_len = len(smooth) if min_len is None else min(min_len, len(smooth))
    date = df["date"].iloc[:min_len].reset_index(drop=True) if "date" in df.columns else pd.Series(range(min_len), name="date")
    out = pd.DataFrame({"date": date})
    for col, bands in temp.items():
        for band, values in bands.items():
            out[f"{col}_{band}"] = values[:min_len]
    return out
