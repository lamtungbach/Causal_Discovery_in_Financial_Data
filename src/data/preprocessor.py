"""
src/data/preprocessor.py
========================
Tính log-return, rolling volatility, chuẩn hóa.

Fix so với phiên bản cũ:
  ✅ Luôn lưu volatility_21d.csv (rolling vol GỐC, annualized)
  ✅ KHÔNG lưu volatility_stationary.csv (đã xóa logic auto-diff)
  ✅ fit_transform() trả về volatility gốc (không phải diff)
  ✅ Xử lý missing bằng ffill trước khi tính return
  ✅ Tương thích với NeuralGrangerCausality
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import (
    LABELS, START_DATE, END_DATE,
    PATH_RAW, PATH_PROCESSED, PATH_RESULTS,
    ROLLING_WINDOW,
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Pipeline xử lý dữ liệu tài chính:
      1. Load prices từ file (hoặc nhận từ bên ngoài)
      2. Tính log-return:           r_t = ln(P_t / P_{t-1})
      3. Tính rolling volatility:   σ_t = std(r) × √252  (window=21 ngày)
      4. Xử lý missing values
      5. Chuẩn hóa StandardScaler

    Interface chính:
        X, vol = prep.fit_transform(prices)   ← dùng trong pipeline/tests
        X, vol, names = prep.run()            ← load file tự động
    """

    def __init__(
        self,
        start        : str            = START_DATE,
        end          : str            = END_DATE,
        window       : int            = ROLLING_WINDOW,
        raw_dir      : str            = PATH_RAW,
        processed_dir: str            = PATH_PROCESSED,
    ):
        self.start         = start
        self.end           = end
        self.window        = window
        self.raw_dir       = raw_dir
        self.processed_dir = processed_dir

        # State — được điền sau mỗi bước
        self.prices_      : Optional[pd.DataFrame] = None
        self.returns_     : Optional[pd.DataFrame] = None
        self.volatility_  : Optional[pd.DataFrame] = None
        self.scale_params_: Optional[dict]         = None

    # ─────────────────────────────────────────────────────────────────────────
    # fit_transform() — Interface chính
    # ─────────────────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        prices     : pd.DataFrame,
        standardize: bool = True,
        save_vol   : bool = True,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Nhận prices → trả về (X_normalized, volatility_df).

        Parameters
        ----------
        prices      : (n_days, n_markets)  giá đóng cửa đã xử lý missing
        standardize : có chuẩn hóa StandardScaler không
        save_vol    : có lưu volatility_21d.csv không

        Returns
        -------
        X   : (n_samples, n_markets)  đã chuẩn hóa, dùng cho NeuralGC/PC/NOTEARS
        vol : pd.DataFrame            volatility gốc (annualized), dùng để plot
        """
        # 1. Xử lý missing trong prices (ffill)
        prices_clean = prices.ffill().bfill()
        self.prices_ = prices_clean

        # 2. Log-return
        self.compute_returns(prices_clean)

        # 3. Rolling volatility (annualized)
        vol = self.compute_volatility()
        vol = vol.ffill().bfill().dropna()

        # Kiểm tra vol không có giá trị âm hoặc inf
        assert (vol >= 0).all().all(), "Volatility có giá trị âm — lỗi dữ liệu!"
        assert np.isfinite(vol.values).all(), "Volatility có inf/NaN — kiểm tra prices!"

        self.volatility_ = vol

        # 4. Lưu volatility gốc (annualized, chưa chuẩn hóa)
        if save_vol:
            os.makedirs(self.processed_dir, exist_ok=True)
            out_path = os.path.join(self.processed_dir, "volatility_21d.csv")
            vol.to_csv(out_path)
            logger.info(f"Đã lưu: {out_path} | shape: {vol.shape}")
            print(f"   ✅ Đã lưu: {out_path} | shape: {vol.shape}")

        # 5. Chuẩn hóa
        if standardize:
            vol_norm, params = self._standardize(vol)
            self.scale_params_ = params
            X = vol_norm.values.astype(np.float32)
        else:
            X = vol.values.astype(np.float32)

        logger.info(f"fit_transform OK | X={X.shape} | vol_range=[{vol.values.min():.4f}, {vol.values.max():.4f}]")
        return X, vol

    # ─────────────────────────────────────────────────────────────────────────
    # Các bước xử lý
    # ─────────────────────────────────────────────────────────────────────────

    def compute_returns(self, prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """r_t = ln(P_t / P_{t-1})"""
        if prices is None:
            prices = self.prices_

        # Kiểm tra giá trị <= 0 trước khi tính log
        n_invalid = (prices <= 0).sum().sum()
        if n_invalid > 0:
            print(f"   ⚠️  Có {n_invalid} giá trị <= 0 trong prices → thay bằng NaN rồi ffill")
            prices = prices.where(prices > 0, other=np.nan).ffill().bfill()

        returns       = np.log(prices / prices.shift(1)).dropna()
        self.returns_ = returns
        return returns

    def compute_volatility(self, returns: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """σ_t = rolling_std(r, window) × √252   (annualized)"""
        if returns is None:
            returns = self.returns_
        vol              = returns.rolling(window=self.window).std() * np.sqrt(252)
        vol              = vol.dropna()
        self.volatility_ = vol
        return vol

    def _standardize(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """(x - mean) / std  per column"""
        means      = data.mean()
        stds       = data.std().replace(0, 1)
        normalized = (data - means) / stds
        params     = {"mean": means.to_dict(), "std": stds.to_dict()}
        return normalized, params

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        """Chuyển X chuẩn hóa về scale gốc (volatility annualized)."""
        if self.scale_params_ is None or self.volatility_ is None:
            raise RuntimeError("Gọi fit_transform(standardize=True) trước.")
        cols  = list(self.volatility_.columns)
        means = np.array([self.scale_params_["mean"][c] for c in cols])
        stds  = np.array([self.scale_params_["std"][c]  for c in cols])
        return pd.DataFrame(X * stds + means, columns=cols)

    # ─────────────────────────────────────────────────────────────────────────
    # run() — Load file và chạy toàn bộ
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, standardize: bool = True):

        # ── Ưu tiên đọc merged_prices.csv ──
        merged_path = os.path.join(self.processed_dir, "merged_prices.csv")
        clean_path  = os.path.join(self.processed_dir, "prices_clean.csv")
        raw_path    = os.path.join(self.raw_dir, "raw_prices.csv")

        if os.path.exists(merged_path):
            print(f"   Load prices từ: {merged_path}")
            prices = pd.read_csv(merged_path, index_col=0, parse_dates=True)
        elif os.path.exists(clean_path):
            print(f"   Load prices từ: {clean_path}")
            prices = pd.read_csv(clean_path, index_col=0, parse_dates=True)
        elif os.path.exists(raw_path):
            print(f"   Load prices từ: {raw_path}")
            prices = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        else:
            raise FileNotFoundError(
                "Không tìm thấy prices file.\n"
                "Chạy: python src/data/merger.py"
            )

        # ── Đảm bảo tên cột khớp với LABELS trong config ──
        prices = prices.rename(columns={"VNIndex": "VNINDEX"})  # phòng trường hợp file cũ

        # Chỉ giữ cột có trong LABELS
        expected = list(LABELS.keys())  # ["SP500", "VNINDEX", "Gold", "Oil", "Bitcoin"]
        prices = prices[[c for c in expected if c in prices.columns]]

        if prices.shape[1] < len(expected):
            missing = set(expected) - set(prices.columns)
            print(f"   ⚠️  Thiếu cột: {missing}")

        prices = prices.loc[self.start:self.end]
        print(f"   Prices shape: {prices.shape}")
        print(f"   Cột: {list(prices.columns)}")

        # Missing check
        miss = prices.isna().sum()
        if miss.sum() > 0:
            print(f"   ⚠️  Missing: {miss[miss>0].to_dict()} → ffill/bfill")

        X, vol = self.fit_transform(prices, standardize=standardize)
        node_names = list(vol.columns)

        print(f"\n   ✅ Preprocessing hoàn tất:")
        print(f"      X shape    : {X.shape}")
        print(f"      X range    : [{X.min():.4f}, {X.max():.4f}]")
        print(f"      Vol range  : [{vol.values.min():.4f}, {vol.values.max():.4f}]")
        print(f"      Node names : {node_names}")
        return X, vol, node_names

    # ─────────────────────────────────────────────────────────────────────────
    # Thống kê mô tả (giữ để backward-compatible)
    # ─────────────────────────────────────────────────────────────────────────

    def describe(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Thống kê mô tả: Mean, Std, Min, Max, Skewness, Kurtosis, JB."""
        try:
            from scipy.stats import jarque_bera, skew, kurtosis
        except ImportError:
            raise ImportError("pip install scipy")

        if data is None:
            data = self.volatility_
        if data is None:
            raise RuntimeError("Gọi fit_transform() hoặc compute_volatility() trước.")

        rows = []
        for col in data.columns:
            s         = data[col].dropna()
            jb_s, jb_p = jarque_bera(s)
            rows.append({
                "Market"    : LABELS.get(col, col),
                "N"         : len(s),
                "Mean"      : round(s.mean(), 6),
                "Std"       : round(s.std(),  6),
                "Min"       : round(s.min(),  6),
                "Max"       : round(s.max(),  6),
                "Skewness"  : round(skew(s),     4),
                "Kurtosis"  : round(kurtosis(s), 4),
                "JB Stat"   : round(jb_s,        2),
                "JB p-value": round(jb_p,        4),
            })
        return pd.DataFrame(rows).set_index("Market")


# ─────────────────────────────────────────────────────────────────────────────
# Run standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("\n📊 DataPreprocessor — Tính Rolling Volatility")
    print("=" * 55)

    prep = DataPreprocessor()
    X, vol, names = prep.run(standardize=True)

    print(f"\n   Volatility (5 hàng đầu):")
    print(vol.head().to_string())
    print(f"\n   X (5 hàng đầu, chuẩn hóa):")
    print(pd.DataFrame(X[:5], columns=names).to_string())