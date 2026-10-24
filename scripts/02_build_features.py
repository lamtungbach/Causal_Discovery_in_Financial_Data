from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import PRICE_FILE, RETURN_FILE, FEATURE_FILE, ASSETS, TABLE_DIR
from src.common import load_timeseries, save_table


def compute_returns_from_price(price):
    df = price[["date"]].copy()
    for key, meta in ASSETS.items():
        col = meta["price"]
        rcol = meta["return"]
        df[rcol] = np.log(price[col] / price[col].shift(1)) * 100.0
    return df.dropna().reset_index(drop=True)


def add_volatility_features(df, windows=(5, 21, 63)):
    out = df.copy()
    for meta in ASSETS.values():
        r = meta["return"]
        label = meta["label"].lower()
        if r not in out: continue
        out[f"abs_{r}"] = out[r].abs()
        for w in windows:
            out[f"vol_{label}_{w}d"] = out[r].rolling(w).std()
    return out


def main():
    if RETURN_FILE.exists():
        ret = load_timeseries(RETURN_FILE)
    elif PRICE_FILE.exists():
        ret = compute_returns_from_price(load_timeseries(PRICE_FILE))
        save_table(ret, RETURN_FILE)
    else:
        raise FileNotFoundError("Cần có data/processed/master_return.csv hoặc master_price.csv")
    features = add_volatility_features(ret)
    features = features.dropna().reset_index(drop=True)
    save_table(features, FEATURE_FILE)
    summary = pd.DataFrame({
        "column": features.columns,
        "dtype": [str(features[c].dtype) for c in features.columns],
        "nan_pct": [features[c].isna().mean() * 100 for c in features.columns],
    })
    save_table(summary, TABLE_DIR / "02_feature_columns.csv")
    print(features.head())

if __name__ == "__main__":
    main()
