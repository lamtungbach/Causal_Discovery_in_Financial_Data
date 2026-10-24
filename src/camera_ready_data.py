from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import (
    ASSETS,
    CAMERA_TRAIN_RATIO,
    CAMERA_VAL_RATIO,
    RAW_DIR,
    RETURN_COLS,
)


RAW_FILES = {
    "vnindex": RAW_DIR / "vnindex_2015_2026.csv",
    "btc": RAW_DIR / "bitcoin_2015_2026.csv",
    "oil": RAW_DIR / "oil_wti_2015_2026.csv",
    "gold": RAW_DIR / "gold_2015_2026.csv",
    "sp500": RAW_DIR / "sp500_2015_2026.csv",
    "sse": RAW_DIR / "sse_2015_2026.csv",
    "vix": RAW_DIR / "vix_2015_2026.csv",
}

PROVIDERS = {
    "vnindex": "VCI via vnstock",
    "btc": "Yahoo Finance via yfinance (auto_adjust=True)",
    "oil": "Yahoo Finance via yfinance (auto_adjust=True)",
    "gold": "Yahoo Finance via yfinance (auto_adjust=True)",
    "sp500": "Yahoo Finance via yfinance (auto_adjust=True)",
    "sse": "Yahoo Finance via yfinance (auto_adjust=True)",
    "vix": "Yahoo Finance via yfinance (auto_adjust=True)",
}


@dataclass
class SupervisedSplit:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    dates_train: np.ndarray
    dates_val: np.ndarray
    dates_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    train_end: int
    val_end: int
    raw_dates: np.ndarray
    raw_returns: np.ndarray
    input_shift_sources: tuple[str, ...]

    @property
    def split_summary(self) -> dict:
        def bounds(dates: np.ndarray) -> tuple[str | None, str | None]:
            if len(dates) == 0:
                return None, None
            return str(pd.Timestamp(dates[0]).date()), str(pd.Timestamp(dates[-1]).date())

        train_start, train_stop = bounds(self.dates_train)
        val_start, val_stop = bounds(self.dates_val)
        test_start, test_stop = bounds(self.dates_test)
        return {
            "raw_observations": len(self.raw_dates),
            "train_forecast_origins": len(self.X_train),
            "validation_forecast_origins": len(self.X_val),
            "test_forecast_origins": len(self.X_test),
            "train_start": train_start,
            "train_end": train_stop,
            "validation_start": val_start,
            "validation_end": val_stop,
            "test_start": test_start,
            "test_end": test_stop,
            "input_shift_sources": ",".join(self.input_shift_sources),
        }


def load_raw_close(asset: str) -> pd.DataFrame:
    path = RAW_FILES[asset]
    if not path.exists():
        raise FileNotFoundError(f"Missing raw input for {asset}: {path}")
    df = pd.read_csv(path, usecols=["date", "close"], parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    return df.rename(columns={"close": ASSETS[asset]["price"]}).reset_index(drop=True)


def build_price_panel(calendar: str = "main") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a price panel while retaining an auditable missing-value report.

    ``main`` reproduces the documented VN-Index-anchored calendar policy.
    ``common`` is the robustness sample: an inner join on actually observed
    dates with no forward filling or interpolation.
    """
    if calendar not in {"main", "common"}:
        raise ValueError("calendar must be 'main' or 'common'")

    frames = {asset: load_raw_close(asset) for asset in RAW_FILES}
    if calendar == "main":
        panel = frames["vnindex"].copy()
        for asset in RAW_FILES:
            if asset != "vnindex":
                panel = panel.merge(frames[asset], on="date", how="left")
    else:
        panel = frames["vnindex"].copy()
        for asset in RAW_FILES:
            if asset != "vnindex":
                panel = panel.merge(frames[asset], on="date", how="inner")

    price_cols = [meta["price"] for meta in ASSETS.values()]
    audit_rows = []
    for col in price_cols:
        audit_rows.append(
            {
                "calendar": calendar,
                "stage": "before_fill",
                "column": col,
                "rows": len(panel),
                "missing": int(panel[col].isna().sum()),
                "non_positive": int((panel[col].dropna() <= 0).sum()),
            }
        )

    if calendar == "main":
        for asset in ("oil", "gold", "sp500", "vix"):
            col = ASSETS[asset]["price"]
            panel[col] = panel[col].ffill(limit=3)
        sse_col = ASSETS["sse"]["price"]
        panel[sse_col] = panel[sse_col].ffill(limit=3).interpolate(
            method="linear", limit=5
        )
        # BTC is intentionally not filled because it trades continuously.

    before_drop = len(panel)
    panel = panel.dropna(subset=price_cols).sort_values("date").reset_index(drop=True)
    for col in price_cols:
        audit_rows.append(
            {
                "calendar": calendar,
                "stage": "after_fill_and_complete_case",
                "column": col,
                "rows": len(panel),
                "missing": int(panel[col].isna().sum()),
                "non_positive": int((panel[col] <= 0).sum()),
            }
        )
    audit_rows.append(
        {
            "calendar": calendar,
            "stage": "rows_removed_for_missing_prices",
            "column": "ALL",
            "rows": before_drop,
            "missing": before_drop - len(panel),
            "non_positive": 0,
        }
    )
    return panel, pd.DataFrame(audit_rows)


def safe_log_return_panel(
    price_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute 100*log returns only when both endpoint prices are positive.

    A non-positive observation invalidates both the return ending on that date
    and the next return. It is not silently bridged by deleting the price row.
    """
    out = price_panel[["date"]].copy()
    invalid_rows = []
    for asset, meta in ASSETS.items():
        price = pd.to_numeric(price_panel[meta["price"]], errors="coerce")
        previous = price.shift(1)
        valid = price.gt(0) & previous.gt(0) & np.isfinite(price) & np.isfinite(previous)
        returns = pd.Series(np.nan, index=price.index, dtype=float)
        returns.loc[valid] = 100.0 * np.log(price.loc[valid] / previous.loc[valid])
        out[meta["return"]] = returns

        affected = price_panel.loc[~valid & (price.index > 0), ["date"]].copy()
        if not affected.empty:
            affected["asset"] = asset
            affected["price_t"] = price.loc[affected.index].to_numpy()
            affected["price_t_minus_1"] = previous.loc[affected.index].to_numpy()
            affected["reason"] = np.where(
                price.loc[affected.index].le(0).to_numpy(),
                "current_price_non_positive",
                np.where(
                    previous.loc[affected.index].le(0).to_numpy(),
                    "previous_price_non_positive",
                    "missing_or_non_finite_endpoint",
                ),
            )
            invalid_rows.append(affected)

    complete = out.dropna(subset=RETURN_COLS).reset_index(drop=True)
    if not np.isfinite(complete[RETURN_COLS].to_numpy()).all():
        raise ValueError("Non-finite values remain after explicit return validation")
    invalid = (
        pd.concat(invalid_rows, ignore_index=True)
        if invalid_rows
        else pd.DataFrame(
            columns=["date", "asset", "price_t", "price_t_minus_1", "reason"]
        )
    )
    return complete, invalid


def build_return_dataset(
    calendar: str = "main",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices, fill_audit = build_price_panel(calendar)
    returns, invalid_returns = safe_log_return_panel(prices)
    return prices, returns, fill_audit, invalid_returns


def prepare_supervised(
    returns: pd.DataFrame,
    lags: int,
    input_shift_sources: Iterable[str] = (),
    train_ratio: float = CAMERA_TRAIN_RATIO,
    val_ratio: float = CAMERA_VAL_RATIO,
) -> SupervisedSplit:
    """Split by target dates, then form lag windows using prior observations.

    Validation and test targets retain their full historical context. Scaling
    statistics are calculated once from raw training-period returns only.
    """
    if lags < 1:
        raise ValueError("lags must be positive")
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be in (0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    df = returns[["date", *RETURN_COLS]].dropna().sort_values("date").reset_index(drop=True)
    raw = df[RETURN_COLS].to_numpy(dtype=np.float64)
    dates = df["date"].to_numpy()
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if train_end <= lags or val_end <= train_end or val_end >= n:
        raise ValueError("Dataset is too short for the requested split and lag length")

    mean = raw[:train_end].mean(axis=0)
    std = raw[:train_end].std(axis=0, ddof=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    scaled = (raw - mean) / std
    model_inputs = scaled.copy()

    shifts = tuple(input_shift_sources)
    unknown = sorted(set(shifts) - set(RETURN_COLS))
    if unknown:
        raise ValueError(f"Unknown shifted source columns: {unknown}")
    for source in shifts:
        j = RETURN_COLS.index(source)
        model_inputs[1:, j] = scaled[:-1, j]
        model_inputs[0, j] = np.nan

    X, Y, target_dates, target_indices = [], [], [], []
    for target_index in range(lags, n):
        window = model_inputs[target_index - lags : target_index]
        target = scaled[target_index]
        if not np.isfinite(window).all() or not np.isfinite(target).all():
            continue
        X.append(window)
        Y.append(target)
        target_dates.append(dates[target_index])
        target_indices.append(target_index)

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    target_dates = np.asarray(target_dates)
    target_indices = np.asarray(target_indices)
    train_mask = target_indices < train_end
    val_mask = (target_indices >= train_end) & (target_indices < val_end)
    test_mask = target_indices >= val_end

    return SupervisedSplit(
        X_train=X[train_mask],
        Y_train=Y[train_mask],
        X_val=X[val_mask],
        Y_val=Y[val_mask],
        X_test=X[test_mask],
        Y_test=Y[test_mask],
        dates_train=target_dates[train_mask],
        dates_val=target_dates[val_mask],
        dates_test=target_dates[test_mask],
        mean=mean,
        std=std,
        train_end=train_end,
        val_end=val_end,
        raw_dates=dates,
        raw_returns=raw,
        input_shift_sources=shifts,
    )


def write_data_audit(
    output_path: Path,
    main_prices: pd.DataFrame,
    main_returns: pd.DataFrame,
    common_returns: pd.DataFrame,
    invalid_returns: pd.DataFrame,
) -> None:
    rows = []
    for asset, path in RAW_FILES.items():
        raw = pd.read_csv(path, parse_dates=["date"])
        rows.append(
            {
                "asset": asset,
                "provider": PROVIDERS[asset],
                "ticker": {
                    "vnindex": "VNINDEX",
                    "btc": "BTC-USD",
                    "oil": "CL=F",
                    "gold": "GC=F",
                    "sp500": "^GSPC",
                    "sse": "000001.SS",
                    "vix": "^VIX",
                }[asset],
                "observations": len(raw),
                "start": str(pd.to_datetime(raw["date"]).min().date()),
                "end": str(pd.to_datetime(raw["date"]).max().date()),
                "minimum_close": float(pd.to_numeric(raw["close"]).min()),
                "non_positive_close": int((pd.to_numeric(raw["close"]) <= 0).sum()),
                "file_timestamp": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
            }
        )

    wti = pd.read_csv(RAW_FILES["oil"], parse_dates=["date"])
    wti_row = wti.loc[wti["date"].eq(pd.Timestamp("2020-04-20"))]
    wti_value = float(wti_row["close"].iloc[0]) if len(wti_row) else float("nan")
    affected_wti = invalid_returns.loc[invalid_returns["asset"].eq("oil")]

    lines = [
        "# Camera-ready data audit",
        "",
        "The exact API retrieval timestamp was not embedded in the CSV files. "
        "The table therefore reports filesystem timestamps and does not present them as API metadata.",
        "",
        "## Provenance and raw coverage",
        "",
        pd.DataFrame(rows).to_markdown(index=False),
        "",
        "## Processing contract",
        "",
        "- Price field: normalized `close`; Yahoo Finance files were downloaded with `auto_adjust=True`.",
        "- Main calendar: VN-Index anchored; OIL/GOLD/SP500/VIX forward-filled up to 3 observations; "
        "SSE forward-filled up to 3 and linearly interpolated up to 5; BTC is not filled.",
        "- Common-day calendar: raw-date intersection across all seven assets; no fill or interpolation.",
        "- Returns: `100 * log(P_t / P_{t-1})` only when both endpoints are finite and strictly positive.",
        "- A non-positive endpoint invalidates the return on that date and the immediately following return; "
        "the price row is not deleted before return construction, so no multi-day bridge is introduced.",
        "",
        "## WTI non-positive-price audit",
        "",
        f"- Raw CL=F close on 2020-04-20: `{wti_value:.6f}`.",
        f"- Explicitly invalidated oil return endpoints in the main calendar: "
        f"`{', '.join(affected_wti['date'].dt.strftime('%Y-%m-%d').tolist())}`.",
        "",
        "## Final samples",
        "",
        f"- Main complete price panel: {len(main_prices):,} rows.",
        f"- Main multivariate return panel: {len(main_returns):,} rows "
        f"({main_returns['date'].min().date()} to {main_returns['date'].max().date()}).",
        f"- Common-day multivariate return panel: {len(common_returns):,} rows "
        f"({common_returns['date'].min().date()} to {common_returns['date'].max().date()}).",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
