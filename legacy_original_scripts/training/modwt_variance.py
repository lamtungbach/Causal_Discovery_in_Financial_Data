"""
07_modwt_variance.py
MODWT Variance Decomposition — phân rã phương sai theo frequency band
Yêu cầu: pip install pywavelets pandas numpy matplotlib seaborn
"""

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ─────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────
DATA_PATH   = 'data/processed/master_return.csv'
OUT_TABLE   = 'results/tables/modwt_variance.csv'
OUT_FIG_BAR = 'results/figures/modwt_variance_barplot.png'
OUT_FIG_HM  = 'results/figures/modwt_variance_heatmap.png'

RETURN_COLS = [
    'return_vnindex', 'return_btc', 'return_oil',
    'return_gold', 'return_sp500', 'return_sse', 'return_vix'
]
LABELS = {
    'return_vnindex': 'VNI',
    'return_btc'    : 'BTC',
    'return_oil'    : 'OIL',
    'return_gold'   : 'GOLD',
    'return_sp500'  : 'SP500',
    'return_sse'    : 'SSE',
    'return_vix'    : 'VIX',
}

PERIODS = {
    'Toàn kỳ (2015–2025)' : ('2015-01-01', '2025-12-31'),
    'Trước COVID (2015–19)': ('2015-01-01', '2019-12-31'),
    'COVID (2020–21)'      : ('2020-01-01', '2021-12-31'),
    'Hậu COVID (2022–25)'  : ('2022-01-01', '2025-12-31'),
}

# MODWT wavelet mẹ — la8 (Least Asymmetric 8) phù hợp tài chính
WAVELET  = 'db4'      # tương đương LA(8) trong R; đổi sang 'sym8' nếu muốn LA(8)
MAX_LEVEL = 8         # số level tối đa (scale 2^1 đến 2^8 ngày)

# Frequency bands theo số ngày (scale)
# Level j tương ứng với chu kỳ [2^j, 2^(j+1)) ngày
BAND_MAP = {
    'Ngắn hạn\n(2–16 ngày)'   : [1, 2, 3],      # level 1–3
    'Trung hạn\n(16–64 ngày)' : [4, 5],          # level 4–5
    'Dài hạn\n(64–256 ngày)'  : [6, 7, 8],       # level 6–8
}

os.makedirs('results/tables',  exist_ok=True)
os.makedirs('results/figures', exist_ok=True)


# ─────────────────────────────────────────────
# HÀM TIỆN ÍCH
# ─────────────────────────────────────────────
def modwt_variance(series: np.ndarray, wavelet: str, level: int) -> dict:
    """
    Tính MODWT variance decomposition.
    Trả về dict: {level_j: variance_j} và 'smooth' cho phần dư (scaling).
    """
    # Dùng SWT (Stationary Wavelet Transform) — tương đương MODWT trong pywt
    coeffs = pywt.swt(series, wavelet=wavelet, level=level, norm=True)
    # coeffs là list các tuple (cA, cD) từ level cao đến thấp
    # cD[j] = detail coefficients tại level j
    variances = {}
    for j, (cA, cD) in enumerate(reversed(coeffs), start=1):
        variances[f'level_{j}'] = float(np.var(cD, ddof=1))
    variances['smooth'] = float(np.var(coeffs[0][0], ddof=1))
    return variances


def variance_to_pct(var_dict: dict, levels: list) -> dict:
    """Chuyển variance tuyệt đối sang % tổng phương sai."""
    total = sum(var_dict.get(f'level_{j}', 0) for j in levels) + var_dict.get('smooth', 0)
    if total == 0:
        return {k: 0.0 for k in var_dict}
    return {k: v / total * 100 for k, v in var_dict.items()}


def assign_band(level_j: int, band_map: dict) -> str:
    for band_name, levels in band_map.items():
        if level_j in levels:
            return band_name
    return 'Khác'


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    df_all = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df_all = df_all.sort_values('date').reset_index(drop=True)

    all_levels = list(range(1, MAX_LEVEL + 1))
    records = []

    for period_name, (start, end) in PERIODS.items():
        df = df_all[(df_all['date'] >= start) & (df_all['date'] <= end)].copy()
        print(f"\n{'='*60}")
        print(f"Giai đoạn: {period_name}  | N = {len(df)}")

        for col in RETURN_COLS:
            series = df[col].dropna().values
            n = len(series)

            # SWT yêu cầu độ dài = bội số của 2^level
            # Cắt bớt về độ dài hợp lệ
            max_lvl = min(MAX_LEVEL, int(np.floor(np.log2(n))))
            padded_n = (n >> max_lvl) << max_lvl  # floor đến bội số 2^max_lvl
            series_trim = series[:padded_n]

            try:
                var_dict = modwt_variance(series_trim, WAVELET, max_lvl)
            except Exception as e:
                print(f"  !! Lỗi {col}: {e}")
                continue

            pct_dict = variance_to_pct(var_dict, list(range(1, max_lvl + 1)))

            # Gộp theo band
            band_pct = {b: 0.0 for b in BAND_MAP}
            band_pct['Smooth (dài hạn+)'] = pct_dict.get('smooth', 0)
            for j in range(1, max_lvl + 1):
                bname = assign_band(j, BAND_MAP)
                if bname in band_pct:
                    band_pct[bname] += pct_dict.get(f'level_{j}', 0)

            row = {
                'Giai đoạn'  : period_name,
                'Biến'       : LABELS[col],
                'N'          : padded_n,
            }
            row.update(band_pct)
            records.append(row)

            print(f"  {LABELS[col]:6s} | " +
                  " | ".join(f"{k.split(chr(10))[0]}: {v:.1f}%" for k, v in band_pct.items()))

    result_df = pd.DataFrame(records)
    result_df.to_csv(OUT_TABLE, index=False, encoding='utf-8-sig')
    print(f"\n✓ Đã lưu bảng: {OUT_TABLE}")

    # ── Vẽ Barplot cho toàn kỳ ──────────────────────────────────────
    df_plot = result_df[result_df['Giai đoạn'] == 'Toàn kỳ (2015–2025)'].copy()
    band_cols = [c for c in df_plot.columns if c not in ('Giai đoạn', 'Biến', 'N')]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_plot))
    width = 0.18
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
    for i, band in enumerate(band_cols):
        ax.bar(x + i * width, df_plot[band].values, width, label=band.replace('\n', ' '), color=colors[i % len(colors)])

    ax.set_xticks(x + width * (len(band_cols) - 1) / 2)
    ax.set_xticklabels(df_plot['Biến'].values, fontsize=11)
    ax.set_ylabel('% Tổng phương sai', fontsize=12)
    ax.set_title('MODWT Variance Decomposition — Toàn kỳ 2015–2025', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG_BAR, dpi=150)
    plt.close()
    print(f"✓ Đã lưu biểu đồ bar: {OUT_FIG_BAR}")

    # ── Vẽ Heatmap: VNI theo từng giai đoạn × band ──────────────────
    df_vni = result_df[result_df['Biến'] == 'VNI'].set_index('Giai đoạn')[band_cols]
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        df_vni.astype(float), annot=True, fmt='.1f', cmap='YlOrRd',
        linewidths=0.5, ax=ax, cbar_kws={'label': '% Phương sai'},
        annot_kws={'size': 10}
    )
    ax.set_title('VNI — % Phương sai theo frequency band & giai đoạn', fontsize=12, fontweight='bold')
    ax.set_xticklabels([c.replace('\n', ' ') for c in band_cols], rotation=15, ha='right')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(OUT_FIG_HM, dpi=150)
    plt.close()
    print(f"✓ Đã lưu heatmap VNI: {OUT_FIG_HM}")

    print("\n✅ MODWT Variance Decomposition hoàn tất!")


if __name__ == '__main__':
    main()