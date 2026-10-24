from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import PRICE_FILE, RETURN_FILE, TABLE_DIR, RAW_DIR, PRICE_COLS, RETURN_COLS
from src.common import load_timeseries, save_table


def profile(df, name, expected_cols):
    rows = []
    rows.append({"file": name, "check": "n_rows", "value": len(df)})
    rows.append({"file": name, "check": "n_cols", "value": df.shape[1]})
    if "date" in df.columns:
        rows.append({"file": name, "check": "date_min", "value": str(pd.to_datetime(df["date"]).min().date())})
        rows.append({"file": name, "check": "date_max", "value": str(pd.to_datetime(df["date"]).max().date())})
        rows.append({"file": name, "check": "duplicate_dates", "value": int(df["date"].duplicated().sum())})
    for c in expected_cols:
        rows.append({"file": name, "check": f"missing_col::{c}", "value": c not in df.columns})
        if c in df.columns:
            rows.append({"file": name, "check": f"nan_pct::{c}", "value": round(df[c].isna().mean() * 100, 4)})
            rows.append({"file": name, "check": f"zero_count::{c}", "value": int((df[c] == 0).sum())})
    return rows


def main():
    records = []
    if PRICE_FILE.exists():
        price = load_timeseries(PRICE_FILE)
        records.extend(profile(price, "master_price.csv", PRICE_COLS))
    else:
        records.append({"file": "master_price.csv", "check": "exists", "value": False})
    if RETURN_FILE.exists():
        ret = load_timeseries(RETURN_FILE)
        records.extend(profile(ret, "master_return.csv", RETURN_COLS))
    else:
        records.append({"file": "master_return.csv", "check": "exists", "value": False})
    raw_files = sorted(p.name for p in RAW_DIR.glob("*.csv"))
    records.append({"file": "raw_dir", "check": "raw_csv_files", "value": ", ".join(raw_files)})
    out = pd.DataFrame(records)
    save_table(out, TABLE_DIR / "01_data_validation_report.csv")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
