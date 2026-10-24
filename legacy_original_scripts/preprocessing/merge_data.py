import numpy as np
import pandas as pd
from pathlib import Path

# ── Đường dẫn ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Mapping: tên asset → tên file raw (không có extension) ─────────────────
#    Điều chỉnh suffix "_2015_2026" nếu bạn dùng suffix khác
FILE_SUFFIX = "_2015_2026"
ASSET_FILES = {
    "vnindex" : RAW_DIR / f"vnindex{FILE_SUFFIX}.csv",
    "btc"     : RAW_DIR / f"bitcoin{FILE_SUFFIX}.csv",
    "oil"     : RAW_DIR / f"oil_wti{FILE_SUFFIX}.csv",
    "gold"    : RAW_DIR / f"gold{FILE_SUFFIX}.csv",
    "sp500"   : RAW_DIR / f"sp500{FILE_SUFFIX}.csv",
    "sse"     : RAW_DIR / f"sse{FILE_SUFFIX}.csv",
    "vix"     : RAW_DIR / f"vix{FILE_SUFFIX}.csv",
}

# ── NaN policy (từ plan.md §3.1) ───────────────────────────────────────────
# assets dùng ffill(limit=3)
SHORT_FILL_ASSETS = ["oil", "gold", "sp500", "vix"]
# SSE: ffill(3) + interpolate(5)
SSE_FILL_ASSET    = "sse"
# BTC: KHÔNG fill — ghi chú trong báo cáo
NO_FILL_ASSET     = "btc"


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def load_asset(name: str, path: Path) -> pd.Series:
    """
    Đọc file CSV, trả về Series close price với index là date (datetime).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"[{name}] Không tìm thấy file: {path}\n"
            f"  → Hãy chạy crawl script trước, sau đó copy/move file vào data/raw/"
        )

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Lấy cột close
    if "close" not in df.columns:
        raise ValueError(f"[{name}] File thiếu cột 'close': {list(df.columns)}")

    series = df.set_index("date")["close"].rename(f"close_{name}")
    print(f"  ✓ {name:<10} {len(series):>5} bản ghi  |  "
          f"{series.index[0].date()} → {series.index[-1].date()}")
    return series


def nan_report(name: str, series: pd.Series, stage: str) -> dict:
    """Trả về thống kê NaN cho một cột tại một giai đoạn."""
    return {
        "asset"  : name,
        "stage"  : stage,
        "total"  : len(series),
        "nan"    : int(series.isna().sum()),
        "pct_nan": round(series.isna().mean() * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*65}")
    print("  BƯỚC 03 — MERGE & PROCESS DATA")
    print(f"{'='*65}\n")

    # ── 1. ĐỌC DỮ LIỆU ────────────────────────────────────────────────────
    print("▶ 1. Đọc dữ liệu raw...")
    series_dict = {}
    for name, path in ASSET_FILES.items():
        series_dict[name] = load_asset(name, path)

    # ── 2. MERGE (VNIndex-anchored left join) ──────────────────────────────
    print(f"\n▶ 2. Merge theo VNIndex-anchored date index...")
    vnindex_series = series_dict["vnindex"]
    df = vnindex_series.reset_index()          # columns: date, close_vnindex
    df = df.rename(columns={"date": "date"})

    for name, s in series_dict.items():
        if name == "vnindex":
            continue
        df = df.merge(s.reset_index(), on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)
    print(f"  → Sau merge: {len(df)} hàng | {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

    # ── 3. BÁO CÁO CHẤT LƯỢNG TRƯỚC KHI XỬ LÝ NaN ────────────────────────
    print(f"\n▶ 3. Kiểm tra NaN trước xử lý...")
    quality_records = []
    price_cols = [f"close_{a}" for a in ASSET_FILES.keys()]

    for col in price_cols:
        name = col.replace("close_", "")
        rec  = nan_report(name, df[col], "before_fill")
        quality_records.append(rec)
        if rec["nan"] > 0:
            print(f"  ⚠  {name:<10} {rec['nan']:>4} NaN ({rec['pct_nan']}%)")
        else:
            print(f"  ✓  {name:<10} 0 NaN")

    # ── 4. XỬ LÝ NaN THEO POLICY ──────────────────────────────────────────
    print(f"\n▶ 4. Xử lý NaN theo policy...")

    # 4a. US-market assets: ffill(limit=3)
    for name in SHORT_FILL_ASSETS:
        col = f"close_{name}"
        before = df[col].isna().sum()
        df[col] = df[col].ffill(limit=3)
        after  = df[col].isna().sum()
        print(f"  ffill(3)            [{name:<6}]  NaN: {before} → {after}")

    # 4b. SSE: ffill(3) + linear interpolate(5)
    col_sse = f"close_{SSE_FILL_ASSET}"
    before  = df[col_sse].isna().sum()
    df[col_sse] = df[col_sse].ffill(limit=3)
    df[col_sse] = df[col_sse].interpolate(method="linear", limit=5)
    after   = df[col_sse].isna().sum()
    print(f"  ffill(3)+interp(5)  [{SSE_FILL_ASSET:<6}]  NaN: {before} → {after}")

    # 4c. BTC: KHÔNG fill — ghi chú
    col_btc = f"close_{NO_FILL_ASSET}"
    n_btc_nan = df[col_btc].isna().sum()
    print(f"  NO FILL             [{NO_FILL_ASSET:<6}]  NaN: {n_btc_nan} (ghi chú báo cáo)")

    # 4d. Drop các hàng còn NaN sau khi fill
    n_before_drop = len(df)
    df_price = df.dropna(subset=price_cols).copy()
    n_dropped = n_before_drop - len(df_price)
    print(f"\n  → Dropped {n_dropped} hàng còn NaN sau khi fill  "
          f"({n_before_drop} → {len(df_price)} hàng)")

    # ── 5. TÍNH LOG-RETURN × 100 ───────────────────────────────────────────
    print(f"\n▶ 5. Tính log-return × 100  [r_t = 100 × ln(P_t / P_t-1)]...")
    df_price = df_price.copy()
    return_cols = []

    for col in price_cols:
        ret_col = col.replace("close_", "return_")
        df_price[ret_col] = 100 * np.log(df_price[col] / df_price[col].shift(1))
        return_cols.append(ret_col)
        print(f"  ✓ {ret_col}")

    # Drop hàng đầu (NaN do shift)
    df_returns = df_price[["date"] + return_cols].dropna().reset_index(drop=True)
    print(f"\n  → master_return: {len(df_returns)} quan sát  |  "
          f"{df_returns['date'].iloc[0].date()} → {df_returns['date'].iloc[-1].date()}")

    # ── 6. BÁO CÁO CHẤT LƯỢNG SAU XỬ LÝ ─────────────────────────────────
    for col in price_cols:
        name = col.replace("close_", "")
        quality_records.append(nan_report(name, df_price[col], "after_fill"))

    # Thêm row về số hàng bị drop
    quality_records.append({
        "asset"  : "ALL",
        "stage"  : "rows_dropped",
        "total"  : n_before_drop,
        "nan"    : n_dropped,
        "pct_nan": round(n_dropped / n_before_drop * 100, 2),
    })

    # ── 7. LƯU FILE ────────────────────────────────────────────────────────
    print(f"\n▶ 6. Lưu file...")

    # master_price.csv
    out_price = PROC_DIR / "master_price.csv"
    df_price[["date"] + price_cols].to_csv(out_price, index=False, encoding="utf-8-sig")
    print(f"  → {out_price}  ({len(df_price)} hàng)")

    # master_return.csv
    out_return = PROC_DIR / "master_return.csv"
    df_returns.to_csv(out_return, index=False, encoding="utf-8-sig")
    print(f"  → {out_return}  ({len(df_returns)} hàng)")

    # data_quality_report.csv
    out_quality = PROC_DIR / "data_quality_report.csv"
    df_quality  = pd.DataFrame(quality_records)
    df_quality.to_csv(out_quality, index=False, encoding="utf-8-sig")
    print(f"  → {out_quality}")

    # ── 8. TÓM TẮT THỐNG KÊ NHANH ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  TÓM TẮT THỐNG KÊ RETURNS")
    print(f"{'='*65}")
    summary = df_returns[return_cols].describe().T[["count", "mean", "std", "min", "max"]]
    summary.index = [c.replace("return_", "").upper() for c in summary.index]
    summary = summary.round(4)
    print(summary.to_string())
    print(f"{'='*65}")
    print("\n✅ Pipeline merge & process hoàn tất!\n")


if __name__ == "__main__":
    main()
