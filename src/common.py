from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def load_timeseries(path: Path, parse_dates: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    df = pd.read_csv(path)
    if parse_dates and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"✓ Saved table: {path}")


def standardize_train_test(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train - mean) / std, (test - mean) / std, mean, std


def split_by_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))].copy()
    return out.reset_index(drop=True)


def make_supervised(values: np.ndarray, target_index: int, lags: int = 5, horizon: int = 1):
    """Tạo dữ liệu supervised: X[t] = các biến từ t-lags+1..t, y = target[t+horizon]."""
    X, y = [], []
    n = len(values)
    for t in range(lags - 1, n - horizon):
        X.append(values[t - lags + 1:t + 1])
        y.append(values[t + horizon, target_index])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
