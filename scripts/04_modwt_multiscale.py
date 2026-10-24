from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import RETURN_FILE, RETURN_COLS, LABELS, TABLE_DIR, PROCESSED_DIR, WAVELET, MAX_LEVEL, BAND_LEVELS, PERIODS
from src.common import load_timeseries, save_table, split_by_period
from src.wavelet_utils import modwt_band_variance, export_decomposed_components


def main():
    df = load_timeseries(RETURN_FILE).dropna().reset_index(drop=True)
    rows = []
    for pname, (start, end) in PERIODS.items():
        sub = split_by_period(df, start, end)
        if len(sub) < 128:
            continue
        for col in RETURN_COLS:
            var_pct = modwt_band_variance(sub[col].values, BAND_LEVELS, wavelet=WAVELET, level=MAX_LEVEL)
            row = {"period": pname, "variable": LABELS.get(col, col), "column": col, "n": len(sub)}
            row.update(var_pct)
            rows.append(row)
    save_table(pd.DataFrame(rows), TABLE_DIR / "04_modwt_band_variance.csv")
    decomp = export_decomposed_components(df, RETURN_COLS, BAND_LEVELS, wavelet=WAVELET, level=MAX_LEVEL)
    save_table(decomp, PROCESSED_DIR / "04_wavelet_decomposed_returns.csv")
    print("✓ MODWT/SWT multi-scale representation exported")

if __name__ == "__main__":
    main()
