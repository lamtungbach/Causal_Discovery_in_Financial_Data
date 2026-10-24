import time
import pandas as pd
from vnstock import Quote

# ── Cấu hình ──────────────────────────────────────────────────────────────
SYMBOL      = "VNINDEX"
SOURCE      = "VCI"        # Nguồn ổn định nhất cho dữ liệu dài hạn
START_YEAR  = 2015
END_YEAR    = 2026         # Cập nhật đến năm 2026 để lấy dữ liệu mới nhất
OUTPUT_FILE = "vnindex_2015_2026.csv"
SLEEP_SEC   = 1.5          # Nghỉ giữa các lần gọi để tránh rate-limit
# ──────────────────────────────────────────────────────────────────────────


def fetch_year(symbol: str, source: str, year: int) -> pd.DataFrame:
    """
    Lấy dữ liệu cho một năm cụ thể.
    Chia nhỏ theo năm để tránh API timeout / dữ liệu bị cắt.
    """
    start = f"{year}-01-01"
    end   = f"{year}-12-31"
    print(f"  Đang lấy năm {year}: {start} → {end} ...", end=" ", flush=True)

    quote = Quote(symbol=symbol, source=source)
    df = quote.history(start=start, end=end, interval="1D")

    if df is None or df.empty:
        print("⚠ Không có dữ liệu!")
        return pd.DataFrame()

    print(f"✓ {len(df)} bản ghi")
    return df


def main():
    print(f"{'='*60}")
    print(f"  Crawl VNIndex | source={SOURCE} | {START_YEAR}–{END_YEAR}")
    print(f"{'='*60}")

    all_frames = []

    for year in range(START_YEAR, END_YEAR + 1):
        try:
            df_year = fetch_year(SYMBOL, SOURCE, year)
            if not df_year.empty:
                all_frames.append(df_year)
        except Exception as e:
            print(f"  ✗ Lỗi năm {year}: {e}")
        time.sleep(SLEEP_SEC)

    if not all_frames:
        print("\n❌ Không crawl được dữ liệu nào. Kiểm tra kết nối / vnstock version.")
        return

    # Ghép và làm sạch
    df_all = pd.concat(all_frames, ignore_index=True)

    # Chuẩn hóa cột thời gian → chỉ giữ phần date
    df_all["time"] = pd.to_datetime(df_all["time"]).dt.normalize()

    # Loại trùng, sắp xếp
    df_all = (
        df_all
        .drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Đổi tên cột 'time' → 'date' cho dễ dùng
    df_all.rename(columns={"time": "date"}, inplace=True)

    print(f"\n{'='*60}")
    print(f"  Tổng cộng : {len(df_all)} phiên giao dịch")
    print(f"  Từ        : {df_all['date'].iloc[0].date()}")
    print(f"  Đến       : {df_all['date'].iloc[-1].date()}")
    print(f"  Cột       : {list(df_all.columns)}")
    print(f"{'='*60}")

    df_all.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ Đã lưu → {OUTPUT_FILE}")

    # Kiểm tra nhanh dữ liệu bị thiếu
    _check_missing(df_all)


def _check_missing(df: pd.DataFrame):
    """In cảnh báo nếu có ngày nào bị NaN trong cột close."""
    n_null = df["close"].isna().sum()
    if n_null > 0:
        print(f"⚠ Cảnh báo: {n_null} bản ghi có giá trị 'close' bị NaN!")
    else:
        print("✓ Không phát hiện giá trị null trong cột 'close'.")


if __name__ == "__main__":
    main()