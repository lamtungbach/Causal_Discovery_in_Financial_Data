from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import RETURN_FILE, RETURN_COLS, LABELS, TABLE_DIR, FIGURE_DIR, PERIODS, TARGET_COL
from src.common import load_timeseries, save_table, split_by_period

PAIRS = [c for c in RETURN_COLS if c != TARGET_COL]


def rolling_coherence_fallback(x, y, windows=(16, 64, 128)):
    # Fallback nhẹ nếu pycwt không có: rolling correlation theo nhiều scale.
    data = {}
    sx = pd.Series(x).reset_index(drop=True)
    sy = pd.Series(y).reset_index(drop=True)
    for w in windows:
        data[f"corr_{w}d"] = sx.rolling(w).corr(sy).abs().values
    return pd.DataFrame(data)


def main():
    df = load_timeseries(RETURN_FILE).dropna().reset_index(drop=True)
    rows = []
    for pname, (start, end) in PERIODS.items():
        sub = split_by_period(df, start, end).dropna()
        if len(sub) < 150:
            continue
        for src in PAIRS:
            fb = rolling_coherence_fallback(sub[src].values, sub[TARGET_COL].values)
            for col in fb.columns:
                rows.append({
                    "period": pname,
                    "source": LABELS.get(src, src),
                    "target": LABELS.get(TARGET_COL, TARGET_COL),
                    "scale_proxy": col,
                    "avg_abs_rolling_corr": float(np.nanmean(fb[col].values)),
                })
            # heatmap fallback
            mat = fb.dropna().T.values
            fig, ax = plt.subplots(figsize=(10, 3))
            im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1)
            ax.set_yticks(range(len(fb.columns)), fb.columns)
            ax.set_title(f"Wavelet-coherence proxy: {LABELS.get(src, src)} → VNI | {pname}")
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            fig.tight_layout()
            fig.savefig(FIGURE_DIR / f"05_wtc_proxy_{LABELS.get(src,src)}_vni_{pname}.png", dpi=150)
            plt.close(fig)
    save_table(pd.DataFrame(rows), TABLE_DIR / "05_wavelet_coherence_proxy_summary.csv")
    print("✓ Wavelet coherence proxy completed. Nếu muốn WTC đúng nghĩa, cài pycwt và thay thế fallback bằng pycwt.wct.")

if __name__ == "__main__":
    main()
