from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import RETURN_FILE, RETURN_COLS, TARGET_COL, TABLE_DIR, MODEL_DIR, DEFAULT_LAGS, SEED
from src.common import load_timeseries, make_supervised, set_seed
from src.metrics import regression_metrics
from src.common import save_table


def main(lags=DEFAULT_LAGS, epochs=40, hidden=16, lr=1e-3, device=None, patience=8):
    import torch
    import torch.nn as nn
    set_seed(SEED)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Bạn chọn device='cuda' nhưng PyTorch không thấy GPU. Kiểm tra torch.cuda.is_available().")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    df = load_timeseries(RETURN_FILE).dropna().reset_index(drop=True)
    values = df[RETURN_COLS].values.astype("float32")
    target_idx = RETURN_COLS.index(TARGET_COL)
    X, y = make_supervised(values, target_idx, lags=lags)
    n = len(X)
    n_train = int(n * 0.7); n_val = int(n * 0.85)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_val], y[n_train:n_val]
    X_test, y_test = X[n_val:], y[n_val:]
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0); std[std == 0] = 1
    X_train = (X_train - mean) / std; X_val = (X_val - mean) / std; X_test = (X_test - mean) / std

    class LSTMRegressor(nn.Module):
        def __init__(self, n_features, hidden):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1]).squeeze(-1)

    print(f"Using device: {device}")
    model = LSTMRegressor(len(RETURN_COLS), hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device); yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device); yv = torch.tensor(y_val, dtype=torch.float32, device=device)
    best, best_state, bad = float("inf"), None, 0
    hist = []
    for ep in range(epochs):
        model.train(); pred = model(Xt); loss = loss_fn(pred, yt)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            val = loss_fn(model(Xv), yv).item()
        hist.append({"epoch": ep+1, "train_loss": float(loss.item()), "val_loss": float(val), "device": device})
        if val < best:
            best = val; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"epoch={ep+1:04d} train={loss.item():.6f} val={val:.6f}")
        if bad >= patience: break
    if best_state: model.load_state_dict(best_state)
    model.to(device); model.eval()
    with torch.no_grad():
        pred_test = model(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()
    metrics = regression_metrics(y_test, pred_test)
    save_table(pd.DataFrame([{"model":"LSTM", "lags":lags, "hidden": hidden, "epochs_requested": epochs, "device": device, **metrics}]), TABLE_DIR / "07_lstm_metrics.csv")
    save_table(pd.DataFrame(hist), TABLE_DIR / "07_lstm_training_history.csv")
    torch.save(model.state_dict(), MODEL_DIR / "07_lstm_vni.pt")
    print("✓ LSTM baseline completed", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lags", type=int, default=DEFAULT_LAGS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    main(lags=args.lags, epochs=args.epochs, hidden=args.hidden, lr=args.lr, device=args.device, patience=args.patience)
