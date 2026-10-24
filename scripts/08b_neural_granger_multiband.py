from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR, RETURN_COLS, LABELS, TABLE_DIR, MODEL_DIR, DEFAULT_LAGS, SEED
from src.common import load_timeseries, make_supervised, set_seed, save_table
from src.neural_granger import LaggedMLPWrapper
from src.metrics import adjacency_metrics, regression_metrics
from scripts import __init__  # noqa: F401


def threshold_adjacency(adj, quantile=0.70):
    x = adj.copy()
    vals = x[x > 0]
    if len(vals) == 0:
        return x
    thr = np.quantile(vals, quantile)
    x[x < thr] = 0.0
    np.fill_diagonal(x, 0.0)
    return x


def run_band(df, band, lags, epochs, lam, hidden, lr, device, threshold_quantile, patience):
    cols = [f"{c}_{band}" for c in RETURN_COLS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột cho band={band}: {missing}")
    values = df[cols].dropna().values.astype("float32")
    X_all, _ = make_supervised(values, 0, lags=lags)
    Y_all = values[lags:]
    n = min(len(X_all), len(Y_all)); X_all, Y_all = X_all[:n], Y_all[:n]
    n_train = int(n * 0.70); n_val = int(n * 0.85)
    X_train, Y_train = X_all[:n_train], Y_all[:n_train]
    X_val, Y_val = X_all[n_train:n_val], Y_all[n_train:n_val]
    X_test, Y_test = X_all[n_val:], Y_all[n_val:]
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0); std[std == 0] = 1
    X_train = (X_train - mean) / std; X_val = (X_val - mean) / std; X_test = (X_test - mean) / std
    Y_mean = Y_train.mean(axis=0, keepdims=True); Y_std = Y_train.std(axis=0, keepdims=True); Y_std[Y_std == 0] = 1
    Y_train_s = (Y_train - Y_mean) / Y_std; Y_val_s = (Y_val - Y_mean) / Y_std

    model = LaggedMLPWrapper(n_vars=len(cols), lags=lags, hidden=hidden, lr=lr, lam=lam, epochs=epochs, patience=patience, seed=SEED, device=device, verbose=True)
    model.fit(X_train, Y_train_s, X_val, Y_val_s)
    adj_raw = model.adjacency()
    adj = threshold_adjacency(adj_raw, threshold_quantile)
    labels = [LABELS[c] for c in RETURN_COLS]
    save_table(pd.DataFrame(adj_raw, index=labels, columns=labels).reset_index().rename(columns={"index":"source"}), TABLE_DIR / f"08b_neural_granger_{band}_adjacency_raw.csv")
    save_table(pd.DataFrame(adj, index=labels, columns=labels).reset_index().rename(columns={"index":"source"}), TABLE_DIR / f"08b_neural_granger_{band}_adjacency_thresholded.csv")
    hist = pd.DataFrame(model.histories)
    hist["band"] = band
    save_table(hist, TABLE_DIR / f"08b_neural_granger_{band}_training_history.csv")
    preds_s = model.predict_all(X_test)
    preds = preds_s * Y_std + Y_mean
    rows = []
    for k, raw_col in enumerate(RETURN_COLS):
        rows.append({"band": band, "target": LABELS[raw_col], **regression_metrics(Y_test[:, k], preds[:, k])})
    save_table(pd.DataFrame(rows), TABLE_DIR / f"08b_neural_granger_{band}_forecast_metrics.csv")
    metric = {**adjacency_metrics(adj), "band": band, "lags": lags, "lambda": lam, "hidden": hidden, "epochs_requested": epochs, "device": model.device}
    np.save(MODEL_DIR / f"08b_neural_granger_{band}_adj.npy", adj)
    return metric


def main(bands=("short", "medium", "long"), lags=DEFAULT_LAGS, epochs=80, lam=1e-3, hidden=32, lr=1e-3, device=None, threshold_quantile=0.70, patience=12):
    set_seed(SEED)
    path = PROCESSED_DIR / "04_wavelet_decomposed_returns.csv"
    if not path.exists():
        raise FileNotFoundError("Chưa có 04_wavelet_decomposed_returns.csv. Hãy chạy scripts/04_modwt_multiscale.py trước.")
    df = load_timeseries(path).dropna().reset_index(drop=True)
    metrics = []
    for band in bands:
        print("\n" + "="*80)
        print(f"RUN MULTI-BAND NEURAL GRANGER: {band}")
        print("="*80)
        metrics.append(run_band(df, band, lags, epochs, lam, hidden, lr, device, threshold_quantile, patience))
    save_table(pd.DataFrame(metrics), TABLE_DIR / "08b_neural_granger_multiband_graph_metrics.csv")
    print("✓ Multi-band Neural Granger completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bands", type=str, default="short,medium,long")
    parser.add_argument("--lags", type=int, default=DEFAULT_LAGS)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--threshold-quantile", type=float, default=0.70)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    bands = tuple(x.strip() for x in args.bands.split(",") if x.strip())
    main(bands=bands, lags=args.lags, epochs=args.epochs, hidden=args.hidden, lr=args.lr, lam=args.lam, device=args.device, threshold_quantile=args.threshold_quantile, patience=args.patience)
