import pandas as pd
from pathlib import Path

# Tự động tính đường dẫn gốc dự án
BASE_DIR = Path(__file__).resolve().parents[2]  # từ src/data/ lên 2 cấp → D:\KLTN
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Đọc file
df_world = pd.read_csv(RAW_DIR / "raw_prices.csv")
df_vn = pd.read_csv(RAW_DIR / "vnindex_manual.csv", usecols=["Date", "Close"])
df_vn.rename(columns={"Close": "VNINDEX"}, inplace=True)

# Đảm bảo cột Date đúng định dạng
df_world["Date"] = pd.to_datetime(df_world["Date"])
df_vn["Date"] = pd.to_datetime(df_vn["Date"])

# Inner Join theo Date
df_merged = pd.merge(df_world, df_vn, on="Date", how="inner")
df_merged = df_merged.sort_values("Date").reset_index(drop=True)

# Kiểm tra
print(f"Shape: {df_merged.shape}")
print(df_merged.head())
print(df_merged.isnull().sum())

# Xuất ra processed
PROCESSED_DIR.mkdir(exist_ok=True)
df_merged.to_csv(PROCESSED_DIR / "merged_prices.csv", index=False)
print(f"Done! Saved to {PROCESSED_DIR / 'merged_prices.csv'}")