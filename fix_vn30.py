"""
fix_vn30.py — Chuẩn hóa file VN30 CSV từ Vietstock.
Chạy: python fix_vn30.py
"""
import pandas as pd
import os

INPUT  = "data/raw/vnindex_manual.csv"
OUTPUT = "data/raw/vnindex_manual.csv"

if not os.path.exists(INPUT):
    # Thử tìm file bất kỳ có chứa "vn30" trong data/raw/
    for f in os.listdir("data/raw"):
        if "vnindex" in f.lower() or "vn30" in f.lower() or "vn-30" in f.lower():
            INPUT = f"data/raw/{f}"
            break
    else:
        print("❌ Không tìm thấy file VN30 trong data/raw/")
        print("   Đảm bảo file đặt đúng vị trí rồi chạy lại.")
        exit(1)

print(f"📄 Đọc file: {INPUT}")

# Đọc file — thử encoding utf-8 rồi latin1
for enc in ("utf-8-sig", "utf-8", "latin1"):
    try:
        df = pd.read_csv(INPUT, encoding=enc)
        break
    except Exception:
        continue

print(f"   Cột hiện tại : {list(df.columns)}")
print(f"   Số dòng      : {len(df)}")
print(f"   5 dòng đầu   :\n{df.head().to_string(index=False)}\n")

# Đổi tên cột về đúng Date, Close nếu cần
df.columns = [c.strip() for c in df.columns]
rename_map = {}
for col in df.columns:
    cl = col.lower()
    if any(x in cl for x in ["date", "ngày", "ngay", "time"]):
        rename_map[col] = "Date"
    elif any(x in cl for x in ["close", "cuối", "cuoi", "đóng", "dong"]):
        rename_map[col] = "Close"
if rename_map:
    df = df.rename(columns=rename_map)

# Chỉ giữ 2 cột Date và Close
df = df[["Date", "Close"]].copy()

# Xóa dấu phẩy trong giá số nếu có: "1,005.19" → 1005.19
df["Close"] = df["Close"].astype(str).str.replace(",", "", regex=False)
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Chuẩn hóa ngày — thử DD/MM/YYYY trước (định dạng Vietstock)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

# Bỏ dòng lỗi
before = len(df)
df = df.dropna(subset=["Date", "Close"])
dropped = before - len(df)
if dropped:
    print(f"   ⚠️  Bỏ {dropped} dòng lỗi (ngày hoặc giá không hợp lệ)")

# Lọc đúng 2018–2024
df = df[(df["Date"] >= "2018-01-01") & (df["Date"] <= "2024-12-31")]

# Sắp xếp tăng dần theo ngày
df = df.sort_values("Date").reset_index(drop=True)

# Lưu — định dạng ngày YYYY-MM-DD, không có dấu phẩy trong số
df.to_csv(OUTPUT, index=False, date_format="%Y-%m-%d")

print(f"""✅ HOÀN THÀNH!

   File lưu tại : {OUTPUT}
   Số ngày      : {len(df)}
   Từ ngày      : {df['Date'].min().date()}
   Đến ngày     : {df['Date'].max().date()}

   5 dòng đầu sau khi sửa:
{df.head().to_string(index=False)}

   5 dòng cuối:
{df.tail().to_string(index=False)}

BƯỚC TIẾP THEO:
   python run_data_pipeline.py
""")
