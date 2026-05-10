"""
=============================================================
  SCRIPT THU THẬP & XỬ LÝ DỮ LIỆU — KHÓA LUẬN TỐT NGHIỆP
  Đề tài: Phân tích Lan tỏa Rủi ro bằng NOTEARS
  Tác giả: [Tên của bạn]
=============================================================

HƯỚNG DẪN CÀI ĐẶT:
    pip install yfinance pandas numpy statsmodels matplotlib seaborn

CÁCH CHẠY:
    python thu_thap_dulieu.py

OUTPUT:
    data/raw_prices.csv          — Giá đóng cửa thô
    data/log_returns.csv         — Log-return hàng ngày
    data/volatility_21d.csv      — Rolling volatility 21 ngày (INPUT cho NOTEARS)
    data/summary_stats.csv       — Thống kê mô tả
    data/adf_kpss_results.csv    — Kết quả kiểm định tính dừng
    figures/volatility_plot.png  — Biểu đồ volatility
    figures/correlation_heatmap.png — Heatmap tương quan
=============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 0. CẤU HÌNH
# ─────────────────────────────────────────────────────────────
START_DATE    = "2018-01-01"
END_DATE      = "2024-12-31"
ROLLING_WINDOW = 21          # ~1 tháng giao dịch
os.makedirs("data",    exist_ok=True)
os.makedirs("figures", exist_ok=True)

TICKERS = {
    "SP500":   "^GSPC",
    "VNIndex": "^VNINDEX",   # Nếu lỗi → xem ghi chú bên dưới
    "Gold":    "GC=F",
    "Oil":     "CL=F",
    "Bitcoin": "BTC-USD",
}

LABELS = {
    "SP500":   "S&P 500",
    "VNIndex": "VN-Index",
    "Gold":    "Vàng (Gold)",
    "Oil":     "Dầu thô (WTI)",
    "Bitcoin": "Bitcoin",
}

COLORS = {
    "SP500":   "#4FC3F7",
    "VNIndex": "#81C784",
    "Gold":    "#FFD700",
    "Oil":     "#FFB74D",
    "Bitcoin": "#FF7043",
}

print("=" * 60)
print("  SCRIPT THU THẬP DỮ LIỆU — NOTEARS THESIS")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# BƯỚC 1: DOWNLOAD DỮ LIỆU GIÁ
# ─────────────────────────────────────────────────────────────
print("\n📥 BƯỚC 1: Đang tải dữ liệu từ Yahoo Finance...")

price_dict = {}
failed = []

for name, ticker in TICKERS.items():
    try:
        print(f"   Đang tải {name} ({ticker})...", end=" ")
        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("Không có dữ liệu")
        price_dict[name] = df["Close"].squeeze()
        print(f"✅ {len(df)} ngày")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        failed.append(name)

# Ghi chú VNIndex
if "VNIndex" in failed:
    print("""
    ⚠️  VNIndex không tải được từ Yahoo Finance.
    👉 Cách lấy thủ công:
       1. Vào https://www.vietstock.vn/VNINDEX/lich-su-gia.htm
       2. Chọn khoảng thời gian 2018-01-01 đến 2024-12-31
       3. Download CSV → đặt tên file: data/vnindex_raw.csv
       4. Đảm bảo có cột 'Date' và 'Close'
       Sau đó chạy lại script.
    """)
    # Thử load từ file thủ công nếu có
    vnindex_path = "data/vnindex_raw.csv"
    if os.path.exists(vnindex_path):
        vn_df = pd.read_csv(vnindex_path, parse_dates=["Date"], index_col="Date")
        price_dict["VNIndex"] = vn_df["Close"]
        print("   ✅ Đã load VNIndex từ file thủ công!")
        failed.remove("VNIndex")

if not price_dict:
    print("❌ Không tải được dữ liệu nào. Kiểm tra kết nối mạng.")
    exit(1)


# ─────────────────────────────────────────────────────────────
# BƯỚC 2: TẠO DATAFRAME GIÁ
# ─────────────────────────────────────────────────────────────
print("\n🔧 BƯỚC 2: Xử lý và căn chỉnh dữ liệu...")

prices = pd.DataFrame(price_dict)
prices.index = pd.to_datetime(prices.index)
prices = prices.sort_index()

# Chỉ giữ ngày có đủ ít nhất 3 thị trường
prices = prices.dropna(thresh=3)

# Forward fill ngày nghỉ lễ (tối đa 3 ngày)
prices = prices.ffill(limit=3)

print(f"   Thời gian: {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"   Số ngày: {len(prices)}")
print(f"   Thị trường: {list(prices.columns)}")
print(f"   Missing values còn lại:\n{prices.isnull().sum().to_string()}")

prices.to_csv("data/raw_prices.csv")
print("   ✅ Đã lưu: data/raw_prices.csv")


# ─────────────────────────────────────────────────────────────
# BƯỚC 3: TÍNH LOG-RETURN
# ─────────────────────────────────────────────────────────────
print("\n📊 BƯỚC 3: Tính log-return...")

log_returns = np.log(prices / prices.shift(1)).dropna()

print(f"   Số ngày sau khi tính return: {len(log_returns)}")
log_returns.to_csv("data/log_returns.csv")
print("   ✅ Đã lưu: data/log_returns.csv")


# ─────────────────────────────────────────────────────────────
# BƯỚC 4: TÍNH ROLLING VOLATILITY (INPUT CHÍNH CHO NOTEARS)
# ─────────────────────────────────────────────────────────────
print(f"\n📈 BƯỚC 4: Tính rolling volatility ({ROLLING_WINDOW} ngày)...")

volatility = log_returns.rolling(window=ROLLING_WINDOW).std()

# Annualize (nhân sqrt(252)) - tùy chọn, comment nếu không cần
# volatility = volatility * np.sqrt(252)

volatility = volatility.dropna()

print(f"   Số ngày sau rolling: {len(volatility)}")
print(f"\n   Thống kê nhanh:")
print(volatility.describe().round(6).to_string())

volatility.to_csv("data/volatility_21d.csv")
print("\n   ✅ Đã lưu: data/volatility_21d.csv  ← FILE NÀY LÀ INPUT CHO NOTEARS")


# ─────────────────────────────────────────────────────────────
# BƯỚC 5: THỐNG KÊ MÔ TẢ
# ─────────────────────────────────────────────────────────────
print("\n📋 BƯỚC 5: Tính thống kê mô tả...")

desc_list = []
for col in volatility.columns:
    s = volatility[col].dropna()
    desc_list.append({
        "Thị trường":   LABELS.get(col, col),
        "Số quan sát":  len(s),
        "Mean":         round(s.mean(), 6),
        "Std":          round(s.std(), 6),
        "Min":          round(s.min(), 6),
        "Max":          round(s.max(), 6),
        "Skewness":     round(stats.skew(s), 4),
        "Kurtosis":     round(stats.kurtosis(s), 4),
        "JB p-value":   round(stats.jarque_bera(s)[1], 4),
    })

desc_df = pd.DataFrame(desc_list).set_index("Thị trường")
print(desc_df.to_string())
desc_df.to_csv("data/summary_stats.csv")
print("\n   ✅ Đã lưu: data/summary_stats.csv")


# ─────────────────────────────────────────────────────────────
# BƯỚC 6: KIỂM ĐỊNH TÍNH DỪNG (ADF & KPSS)
# ─────────────────────────────────────────────────────────────
print("\n🔬 BƯỚC 6: Kiểm định tính dừng ADF & KPSS...")
print(f"   {'Thị trường':<12} {'ADF stat':>10} {'ADF p-val':>10} {'ADF kết quả':>14} {'KPSS stat':>10} {'KPSS p-val':>10} {'KPSS kết quả':>14}")
print("   " + "-" * 82)

adf_results = []
for col in volatility.columns:
    s = volatility[col].dropna()

    # ADF test: H0 = phi dừng. p < 0.05 → bác bỏ H0 → DỪng
    adf_stat, adf_p, _, _, _, _ = adfuller(s, autolag='AIC')
    adf_ok = adf_p < 0.05

    # KPSS test: H0 = dừng. p < 0.05 → bác bỏ H0 → PHI dừng
    try:
        kpss_stat, kpss_p, _, _ = kpss(s, regression='c', nlags='auto')
        kpss_ok = kpss_p >= 0.05
    except Exception:
        kpss_stat, kpss_p, kpss_ok = None, None, None

    label = LABELS.get(col, col)
    adf_label  = "✅ Dừng"    if adf_ok    else "❌ Phi dừng"
    kpss_label = "✅ Dừng"    if kpss_ok   else "❌ Phi dừng"
    if kpss_ok is None:
        kpss_stat_str = kpss_p_str = kpss_label = "N/A"
    else:
        kpss_stat_str = f"{kpss_stat:.4f}"
        kpss_p_str    = f"{kpss_p:.4f}"

    print(f"   {label:<12} {adf_stat:>10.4f} {adf_p:>10.4f} {adf_label:>14} "
          f"{kpss_stat_str:>10} {kpss_p_str:>10} {kpss_label:>14}")

    adf_results.append({
        "Thị trường":   label,
        "ADF stat":     round(adf_stat, 4),
        "ADF p-value":  round(adf_p, 4),
        "ADF kết quả":  "Dừng" if adf_ok else "Phi dừng",
        "KPSS stat":    round(kpss_stat, 4) if kpss_stat else None,
        "KPSS p-value": round(kpss_p, 4)    if kpss_p   else None,
        "KPSS kết quả": "Dừng" if kpss_ok  else "Phi dừng",
        "Kết luận":     "Dừng" if (adf_ok and kpss_ok) else "Phi dừng",
    })

adf_df = pd.DataFrame(adf_results).set_index("Thị trường")
adf_df.to_csv("data/adf_kpss_results.csv")
print("\n   ✅ Đã lưu: data/adf_kpss_results.csv")

# Cảnh báo nếu có chuỗi phi dừng
phi_dung = adf_df[adf_df["Kết luận"] == "Phi dừng"].index.tolist()
if phi_dung:
    print(f"\n   ⚠️  CÁC CHUỖI PHI DỪNG: {phi_dung}")
    print("   → Cần lấy sai phân bậc 1 trước khi đưa vào NOTEARS!")
    print("   → Xem phần xử lý phi dừng ở cuối script.")


# ─────────────────────────────────────────────────────────────
# BƯỚC 7: VẼ BIỂU ĐỒ
# ─────────────────────────────────────────────────────────────
print("\n🎨 BƯỚC 7: Vẽ biểu đồ...")

# --- Biểu đồ 1: Volatility theo thời gian ---
fig = plt.figure(figsize=(16, 14), facecolor="#0F1117")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
cols = list(volatility.columns)

for i, col in enumerate(cols):
    if i >= 5:
        break
    ax = fig.add_subplot(gs[i // 2, i % 2])
    c  = COLORS.get(col, "#AAAAAA")
    ax.plot(volatility.index, volatility[col], color=c, linewidth=0.8)
    ax.fill_between(volatility.index, volatility[col], alpha=0.12, color=c)
    # Đánh dấu cú sốc
    ax.axvline(pd.Timestamp("2020-03-15"), color="white", ls="--", lw=0.7, alpha=0.6, label="COVID")
    ax.axvline(pd.Timestamp("2022-06-01"), color="#FF5252", ls="--", lw=0.7, alpha=0.6, label="Crypto crash")
    ax.set_facecolor("#1A1D2E")
    ax.set_title(LABELS.get(col, col), color="white", fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(colors="#888888", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

# Panel tổng hợp
ax6 = fig.add_subplot(gs[2, :])
ax6.set_facecolor("#1A1D2E")
for col in cols:
    v = volatility[col]
    v_norm = (v - v.mean()) / v.std()
    ax6.plot(volatility.index, v_norm, color=COLORS.get(col, "#AAAAAA"),
             linewidth=0.9, label=LABELS.get(col, col), alpha=0.85)
ax6.axvline(pd.Timestamp("2020-03-15"), color="white", ls="--", lw=0.8, alpha=0.5, label="COVID-19")
ax6.axvline(pd.Timestamp("2022-06-01"), color="#FF5252", ls="--", lw=0.8, alpha=0.5, label="Crypto crash")
ax6.legend(loc="upper left", fontsize=8, framealpha=0.3, labelcolor="white", facecolor="#222244")
ax6.set_title("So sánh Volatility chuẩn hóa — Tất cả thị trường", color="white", fontsize=11, fontweight="bold")
ax6.tick_params(colors="#888888", labelsize=7)
for sp in ax6.spines.values():
    sp.set_edgecolor("#333355")

fig.suptitle(f"Rolling Volatility {ROLLING_WINDOW} ngày — {START_DATE} đến {END_DATE}",
             color="white", fontsize=13, fontweight="bold", y=0.99)
plt.savefig("figures/volatility_plot.png", dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.close()
print("   ✅ Đã lưu: figures/volatility_plot.png")

# --- Biểu đồ 2: Correlation Heatmap ---
fig2, ax2 = plt.subplots(figsize=(8, 6), facecolor="#1A1D2E")
ax2.set_facecolor("#1A1D2E")
corr = volatility.corr()
corr.index   = [LABELS.get(c, c) for c in corr.index]
corr.columns = [LABELS.get(c, c) for c in corr.columns]
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="#333355",
            ax=ax2, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 10, "color": "white"})
ax2.set_title("Correlation Matrix — Volatility 21 ngày", color="white", fontsize=12, fontweight="bold", pad=12)
ax2.tick_params(colors="white", labelsize=9)
plt.tight_layout()
plt.savefig("figures/correlation_heatmap.png", dpi=150, bbox_inches="tight", facecolor="#1A1D2E")
plt.close()
print("   ✅ Đã lưu: figures/correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────
# BƯỚC 8: XỬ LÝ PHI DỪNG (nếu cần)
# ─────────────────────────────────────────────────────────────
if phi_dung:
    print("\n🔧 BƯỚC 8: Lấy sai phân bậc 1 cho chuỗi phi dừng...")
    vol_stationary = volatility.copy()
    for col in volatility.columns:
        label = LABELS.get(col, col)
        if label in phi_dung:
            vol_stationary[col] = volatility[col].diff()
            print(f"   Đã lấy sai phân: {label}")
    vol_stationary = vol_stationary.dropna()
    vol_stationary.to_csv("data/volatility_stationary.csv")
    print("   ✅ Đã lưu: data/volatility_stationary.csv  ← Dùng file này cho NOTEARS nếu có phi dừng")
else:
    print("\n✅ BƯỚC 8: Tất cả chuỗi đã dừng — KHÔNG cần lấy sai phân!")
    print("   → Dùng trực tiếp data/volatility_21d.csv cho NOTEARS")


# ─────────────────────────────────────────────────────────────
# TỔNG KẾT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  HOÀN THÀNH! CÁC FILE ĐÃ TẠO:")
print("=" * 60)
print("  data/")
print("    ├── raw_prices.csv          ← Giá thô")
print("    ├── log_returns.csv         ← Log-return")
print("    ├── volatility_21d.csv      ← INPUT CHÍNH cho NOTEARS")
print("    ├── summary_stats.csv       ← Thống kê mô tả (Chương 4.1)")
print("    └── adf_kpss_results.csv    ← Kết quả ADF/KPSS (Chương 3.2)")
print("  figures/")
print("    ├── volatility_plot.png     ← Biểu đồ cho Chương 4.1")
print("    └── correlation_heatmap.png ← Heatmap tương quan")
print()
print("  BƯỚC TIẾP THEO:")
print("  → Chạy script NOTEARS: python chay_notears.py")
print("=" * 60)
