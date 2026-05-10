"""
src/data/downloader.py
======================
Tải dữ liệu giá từ Yahoo Finance + VNIndex thủ công.

Fix so với phiên bản cũ:
  ✅ Align tất cả markets về business days chung (inner join → forward fill)
  ✅ VNIndex: forward-fill ngày nghỉ VN (thứ 7, CN, lễ) thay vì để NaN
  ✅ Không drop ngày chỉ vì 1 market nghỉ
  ✅ Lưu thêm prices_clean.csv (sau khi xử lý missing)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import TICKERS, LABELS, START_DATE, END_DATE, PATH_RAW, PATH_PROCESSED

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False
    print("⚠️  yfinance chưa cài. Chạy: pip install yfinance")


# ─────────────────────────────────────────────────────────────────────────────
# Download từng market
# ─────────────────────────────────────────────────────────────────────────────

def download_single(name: str, ticker: str) -> pd.Series | None:
    """Tải giá đóng cửa 1 market từ Yahoo Finance."""
    if not HAS_YF:
        return None
    try:
        print(f"   Đang tải {name} ({ticker})...", end=" ", flush=True)
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            raise ValueError("Không có dữ liệu")
        series = df["Close"].squeeze()
        series.name = name
        series.index = pd.to_datetime(series.index)
        print(f"✅ {len(series)} ngày")
        return series
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None


def load_vnindex_manual() -> pd.Series | None:
    """
    Load VNIndex từ CSV thủ công.

    Định dạng file chấp nhận:
      - Có cột 'Date' (hoặc index) và một trong: 'Close', 'close', 'Đóng cửa'
      - Ngày định dạng YYYY-MM-DD hoặc DD/MM/YYYY

    Nguồn tải:
      https://www.vietstock.vn/VNINDEX/lich-su-gia.htm
      Lưu vào: data/raw/vnindex_manual.csv
    """
    path = os.path.join(PATH_RAW, "vnindex_manual.csv")
    if not os.path.exists(path):
        print(f"   ⚠️  Không tìm thấy {path}")
        return None

    try:
        df = pd.read_csv(path)

        # Tìm cột date
        date_col = None
        for c in df.columns:
            if c.strip().lower() in ("date", "ngày", "time", "tradingdate"):
                date_col = c
                break
        if date_col is None:
            # Thử index
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, dayfirst=True)
        else:
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
            df = df.set_index(date_col)
            df.index = pd.to_datetime(df.index)

        # Tìm cột close
        close_col = None
        for c in df.columns:
            if c.strip().lower() in ("close", "đóng cửa", "gia_dong_cua", "closeindex"):
                close_col = c
                break
        if close_col is None:
            close_col = df.columns[0]

        series = df[close_col].squeeze().astype(float)
        series.name = "VNIndex"
        series = series.sort_index()
        series = series.loc[START_DATE:END_DATE]

        print(f"   ✅ VNIndex thủ công: {len(series)} ngày "
              f"({series.index[0].date()} → {series.index[-1].date()})")
        return series

    except Exception as e:
        print(f"   ❌ Lỗi đọc vnindex_manual.csv: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Align & xử lý missing
# ─────────────────────────────────────────────────────────────────────────────

def align_and_fill(price_dict: dict) -> pd.DataFrame:
    """
    Gộp tất cả series thành DataFrame, xử lý missing đúng cách.

    Chiến lược:
      1. Outer join (giữ TẤT CẢ ngày của bất kỳ market nào)
      2. Forward-fill trong mỗi cột (ngày nghỉ = giá ngày trước)
      3. Backward-fill để xử lý đầu chuỗi
      4. Chỉ drop ngày mà TẤT CẢ markets đều NaN
    """
    # Outer join
    prices = pd.concat(price_dict.values(), axis=1, join="outer")
    prices.columns = list(price_dict.keys())
    prices.index   = pd.to_datetime(prices.index)
    prices         = prices.sort_index()

    # Lọc theo khoảng thời gian
    prices = prices.loc[START_DATE:END_DATE]

    # Thống kê missing trước khi fill
    missing_before = prices.isna().sum()
    print(f"\n   Missing trước fill:")
    for col in prices.columns:
        if missing_before[col] > 0:
            pct = missing_before[col] / len(prices) * 100
            print(f"     {col:<12}: {missing_before[col]:>4} ngày ({pct:.1f}%)")

    # Forward-fill rồi backward-fill
    prices = prices.ffill().bfill()

    # Drop ngày mà tất cả đều NaN (trường hợp hiếm)
    prices = prices.dropna(how="all")

    missing_after = prices.isna().sum().sum()
    if missing_after == 0:
        print(f"   ✅ Sau fill: không còn NaN")
    else:
        print(f"   ⚠️  Sau fill: còn {missing_after} NaN")

    return prices


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def download_all() -> pd.DataFrame:
    """
    Tải và xử lý toàn bộ dữ liệu giá.

    Returns
    -------
    prices : pd.DataFrame  giá đóng cửa đã xử lý missing
    """
    print("\n📥 Đang tải dữ liệu thị trường tài chính...")
    print(f"   Giai đoạn: {START_DATE} → {END_DATE}")
    os.makedirs(PATH_RAW,       exist_ok=True)
    os.makedirs(PATH_PROCESSED, exist_ok=True)

    price_dict = {}

    for name, ticker in TICKERS.items():
        # VNIndex: ưu tiên file thủ công
        if name == "VNIndex":
            manual = load_vnindex_manual()
            if manual is not None:
                price_dict["VNIndex"] = manual
                continue

        series = download_single(name, ticker)
        if series is not None:
            price_dict[name] = series
        else:
            print(f"   ⚠️  Bỏ qua {name} — không tải được")

    if not price_dict:
        raise RuntimeError("❌ Không tải được dữ liệu nào!")

    # Align và xử lý missing
    prices = align_and_fill(price_dict)

    # ── Lưu file ──────────────────────────────────────────────────────────────
    raw_path   = os.path.join(PATH_RAW,       "raw_prices.csv")
    clean_path = os.path.join(PATH_PROCESSED, "prices_clean.csv")

    prices.to_csv(raw_path)
    prices.to_csv(clean_path)

    print(f"\n   📁 Đã lưu:")
    print(f"      {raw_path}")
    print(f"      {clean_path}")
    print(f"\n   📊 Tổng kết:")
    print(f"      Thời gian : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"      Số ngày   : {len(prices)}")
    print(f"      Thị trường: {list(prices.columns)}")
    print(f"\n   Giá cuối cùng (hàng cuối):")
    print(prices.tail(3).to_string())

    return prices


if __name__ == "__main__":
    download_all()