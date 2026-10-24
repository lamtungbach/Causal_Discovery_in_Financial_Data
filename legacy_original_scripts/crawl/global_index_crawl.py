import time
import pandas as pd
import yfinance as yf

# ── Cấu hình ──────────────────────────────────────────────────────────────
START_DATE  = "2015-01-01"
END_DATE    = "2026-05-12"  # Cập nhật đến ngày hiện tại để lấy dữ liệu 2026
SLEEP_SEC   = 2.0           # Nghỉ giữa các lần gọi để tránh rate-limit
OUTPUT_DIR  = "."           # Thư mục lưu file CSV (thay đổi nếu cần)

# Danh sách ticker và tên file tương ứng
TICKERS = {
    "bitcoin"  : "BTC-USD",
    "oil_wti"  : "CL=F",
    "gold"     : "GC=F",
    "sp500"    : "^GSPC",
    "sse"      : "000001.SS",
    "vix"      : "^VIX",     # [MỚI] CBOE Volatility Index — Fear Index
}

# Cột OHLCV cần giữ lại (yfinance trả về multi-level index khi multi-ticker)
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
# ──────────────────────────────────────────────────────────────────────────


def fetch_ticker(name: str, ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download dữ liệu lịch sử cho một ticker từ Yahoo Finance.
    Trả về DataFrame chuẩn hóa với cột: date, open, high, low, close, volume.
    """
    print(f"  [{name.upper()}] {ticker}: {start} → {end} ...", end=" ", flush=True)

    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,   # Giá đã điều chỉnh (khuyến nghị cho chuỗi dài)
        progress=False,
    )

    if df is None or df.empty:
        print("⚠ Không có dữ liệu!")
        return pd.DataFrame()

    # yfinance đôi khi trả về MultiIndex columns khi chỉ có 1 ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Chỉ giữ cột cần thiết (một số ticker thiếu Volume)
    cols_available = [c for c in OHLCV_COLS if c in df.columns]
    df = df[cols_available].copy()

    # Chuẩn hóa tên cột → chữ thường
    df.columns = [c.lower() for c in df.columns]

    # Reset index để 'Date' thành cột thường
    df = df.reset_index()
    df.rename(columns={"Date": "date", "index": "date"}, inplace=True)

    # Đảm bảo cột date là datetime, không có phần giờ
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Lọc đúng khoảng thời gian (yfinance có thể trả thêm ngày ngoài range)
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    # Sắp xếp và loại trùng
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    print(f"✓ {len(df)} bản ghi  |  "
          f"{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    return df


def check_quality(name: str, df: pd.DataFrame):
    """Kiểm tra nhanh chất lượng dữ liệu sau khi crawl."""
    if df.empty:
        return
    n_null = df["close"].isna().sum()
    if n_null > 0:
        print(f"    ⚠ [{name}] {n_null} bản ghi có 'close' = NaN — cần xử lý!")
    else:
        print(f"    ✓ [{name}] Không có null trong cột 'close'.")


def main():
    print(f"{'='*65}")
    print(f"  Crawl Global Indices | {START_DATE} → {END_DATE}")
    print(f"  Tickers: {', '.join(TICKERS.values())}")
    print(f"{'='*65}\n")

    summary = []

    for name, ticker in TICKERS.items():
        try:
            df = fetch_ticker(name, ticker, START_DATE, END_DATE)

            if not df.empty:
                # Kiểm tra chất lượng
                check_quality(name, df)

                # Lưu ra file riêng
                out_path = f"{OUTPUT_DIR}/{name}_2015_2026.csv"
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
                print(f"    → Đã lưu: {out_path}\n")

                summary.append({
                    "name"    : name,
                    "ticker"  : ticker,
                    "records" : len(df),
                    "from"    : str(df["date"].iloc[0].date()),
                    "to"      : str(df["date"].iloc[-1].date()),
                    "file"    : out_path,
                })
            else:
                summary.append({
                    "name": name, "ticker": ticker,
                    "records": 0, "from": "N/A", "to": "N/A", "file": "FAILED",
                })

        except Exception as e:
            print(f"    ✗ Lỗi [{name}] {ticker}: {e}\n")
            summary.append({
                "name": name, "ticker": ticker,
                "records": 0, "from": "N/A", "to": "N/A", "file": f"ERROR: {e}",
            })

        time.sleep(SLEEP_SEC)

    # In bảng tóm tắt
    print(f"\n{'='*65}")
    print("  TÓM TẮT KẾT QUẢ")
    print(f"{'='*65}")
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    print(f"{'='*65}")

    # Lưu summary
    summary_path = f"{OUTPUT_DIR}/crawl_summary.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Đã lưu báo cáo tổng hợp → {summary_path}")


if __name__ == "__main__":
    main()