from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import RETURN_FILE, RETURN_COLS, LABELS, TABLE_DIR, MODEL_DIR, DEFAULT_LAGS, SEED
from src.common import load_timeseries, make_supervised, set_seed, save_table
from src.neural_granger import LaggedMLPWrapper
from src.metrics import adjacency_metrics, regression_metrics


def threshold_adjacency(adj, quantile=0.70):
    x = adj.copy()
    vals = x[x > 0]
    if len(vals) == 0:
        return x
    thr = np.quantile(vals, quantile)
    x[x < thr] = 0.0
    np.fill_diagonal(x, 0.0)
    return x


def main(lags=DEFAULT_LAGS, epochs=60, lam=1e-3, hidden=16, lr=1e-3, device=None, threshold_quantile=0.70, patience=8):
    set_seed(SEED)
    df = load_timeseries(RETURN_FILE).dropna().reset_index(drop=True)
    values = df[RETURN_COLS].values.astype("float32")
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

    model = LaggedMLPWrapper(
        n_vars=len(RETURN_COLS), lags=lags, hidden=hidden, lr=lr, lam=lam,
        epochs=epochs, patience=patience, seed=SEED, device=device, verbose=True
    )
    model.fit(X_train, Y_train_s, X_val, Y_val_s)
    adj_raw = model.adjacency()
    adj = threshold_adjacency(adj_raw, quantile=threshold_quantile)
    labels = [LABELS[c] for c in RETURN_COLS]
    save_table(pd.DataFrame(adj_raw, index=labels, columns=labels).reset_index().rename(columns={"index":"source"}), TABLE_DIR / "08_neural_granger_adjacency_raw.csv")
    save_table(pd.DataFrame(adj, index=labels, columns=labels).reset_index().rename(columns={"index":"source"}), TABLE_DIR / "08_neural_granger_adjacency_thresholded.csv")
    save_table(pd.DataFrame(model.histories), TABLE_DIR / "08_neural_granger_training_history.csv")
    preds_s = model.predict_all(X_test)
    preds = preds_s * Y_std + Y_mean
    rows = []
    for k, col in enumerate(RETURN_COLS):
        rows.append({"target": LABELS[col], **regression_metrics(Y_test[:, k], preds[:, k])})
    save_table(pd.DataFrame(rows), TABLE_DIR / "08_neural_granger_forecast_metrics.csv")
    save_table(pd.DataFrame([{**adjacency_metrics(adj), "lags":lags, "lambda":lam, "hidden": hidden, "epochs_requested": epochs, "threshold_quantile": threshold_quantile, "device": model.device}]), TABLE_DIR / "08_neural_granger_graph_metrics.csv")
    np.save(MODEL_DIR / "08_neural_granger_adj.npy", adj)
    print("✓ Neural Granger completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lags", type=int, default=DEFAULT_LAGS)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--threshold-quantile", type=float, default=0.70)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    main(lags=args.lags, epochs=args.epochs, hidden=args.hidden, lr=args.lr, lam=args.lam, device=args.device, threshold_quantile=args.threshold_quantile, patience=args.patience)
