# Kế hoạch Khóa luận Tốt nghiệp — Phiên bản 2.0

## Mối quan hệ giữa tỷ suất lợi nhuận của Bitcoin, Vàng, Dầu thô, S&P 500, SSE và VN-Index: Tiếp cận bằng Wavelet Coherence

> **Định hướng:** Mở rộng từ Bùi Thị Thu Thảo & Cao Tấn Huy (2025) — bổ sung S&P 500, SSE Composite, VIX; tích hợp Wavelet-Granger Causality, MODWT Variance Decomposition và Event Window Analysis  
> **Giai đoạn:** 2015-01-01 → 2025-12-31 · **Tần suất:** ngày (daily)  
> **Phương pháp chính:** Wavelet Coherence (WTC) + MODWT + Wavelet-Granger Causality  
> **Cập nhật:** 2026

---

## Mục lục

1. [Cấu trúc bài khóa luận](#1-cấu-trúc-bài-khóa-luận)
2. [Chương 1 — Giới thiệu](#2-chương-1--giới-thiệu)
3. [Chương 2 — Tổng quan lý thuyết & nghiên cứu thực nghiệm](#3-chương-2--tổng-quan-lý-thuyết--nghiên-cứu-thực-nghiệm)
4. [Chương 3 — Dữ liệu & Phương pháp nghiên cứu](#4-chương-3--dữ-liệu--phương-pháp-nghiên-cứu)
5. [Chương 4 — Kết quả & Thảo luận](#5-chương-4--kết-quả--thảo-luận)
6. [Chương 5 — Kết luận & Khuyến nghị](#6-chương-5--kết-luận--khuyến-nghị)
7. [Pipeline dữ liệu chi tiết](#7-pipeline-dữ-liệu-chi-tiết)
8. [Checklist thực hiện](#8-checklist-thực-hiện)
9. [Tài liệu tham khảo](#9-tài-liệu-tham-khảo)

---

## 1. Cấu trúc bài khóa luận

```
Chương 1: Giới thiệu
  1.1 Bối cảnh & tính cấp thiết
  1.2 Mục tiêu & câu hỏi nghiên cứu
  1.3 Đối tượng & phạm vi
  1.4 Đóng góp của nghiên cứu
  1.5 Cấu trúc bài

Chương 2: Tổng quan lý thuyết & nghiên cứu thực nghiệm
  2.1 Cơ sở lý thuyết
  2.2 Nghiên cứu thực nghiệm
  2.3 Khoảng trống & điểm mới

Chương 3: Dữ liệu & Phương pháp nghiên cứu
  3.1 Dữ liệu nghiên cứu
  3.2 Phương pháp Wavelet Coherence (WTC)
  3.3 [MỚI] MODWT Variance Decomposition
  3.4 [MỚI] Wavelet-Granger Causality
  3.5 [MỚI] Event Window Analysis

Chương 4: Kết quả & Thảo luận
  4.1 Thống kê mô tả
  4.2 Ma trận tương quan Pearson
  4.3 Kiểm định tính dừng (ADF)
  4.4 [MỚI] MODWT — Phân rã phương sai theo frequency band
  4.5 Kết quả Wavelet Coherence — theo từng giai đoạn
  4.6 [MỚI] Wavelet-Granger Causality — theo frequency band
  4.7 [MỚI] Event Window Analysis — 3 sự kiện cụ thể
  4.8 [MỚI] Average Coherence Heatmap — tổng hợp toàn bộ
  4.9 Tổng hợp so sánh

Chương 5: Kết luận & Khuyến nghị
  5.1 Kết luận
  5.2 Khuyến nghị
  5.3 Hạn chế & hướng nghiên cứu tiếp theo

Tài liệu tham khảo
Phụ lục
```

---

## 2. Chương 1 — Giới thiệu

### 1.1 Bối cảnh & tính cấp thiết

**Luận điểm chính cần xây dựng:**

- Thị trường tài chính toàn cầu trải qua nhiều cú sốc hệ thống trong thập niên 2015–2025: COVID-19 (2020–2021), xung đột Nga–Ukraine (2022), biến động tiền điện tử (2017, 2021, 2022), căng thẳng thương mại Mỹ–Trung.
- Thị trường chứng khoán Việt Nam (VN-Index) ngày càng hội nhập sâu → nhạy cảm hơn với biến động toàn cầu.
- Mối quan hệ giữa các tài sản không cố định mà biến đổi theo thời gian và tần số — các phương pháp tuyến tính truyền thống (Pearson, GARCH) không nắm bắt được đặc điểm này.
- Wavelet Coherence là phương pháp phù hợp nhất để phân tích đồng thời miền thời gian và tần số.

**Điểm mới so với Bùi Thị Thu Thảo & Cao Tấn Huy (2025):**

| Khía cạnh | Bài báo gốc (2025) | Khóa luận này |
|---|---|---|
| Biến nghiên cứu | BTC, Vàng, Dầu, VNI (4 biến) | + S&P 500, SSE, **VIX** (7 biến) |
| Giai đoạn | 2015–2023 | 2015–2025 (bổ sung hậu Nga–Ukraine) |
| Phương pháp | WTC | WTC + **MODWT** + **Wavelet-Granger** |
| Phân tích sự kiện | Không | **Event Window Analysis** (3 sự kiện) |
| Trình bày kết quả | Biểu đồ WTC | + **Average Coherence Heatmap** + Bảng định lượng |

### 1.2 Mục tiêu & câu hỏi nghiên cứu

**Mục tiêu tổng quát:** Phân tích mối quan hệ động theo thời gian và tần số giữa tỷ suất lợi nhuận của các tài sản toàn cầu (Bitcoin, Vàng, Dầu thô, S&P 500, SSE, VIX) và VN-Index trong giai đoạn 2015–2025.

**Câu hỏi nghiên cứu:**

1. Mức độ liên kết giữa VN-Index với từng tài sản toàn cầu thay đổi như thế nào theo thời gian và chu kỳ dao động?
2. Tài sản nào đóng vai trò dẫn dắt (leading) hay theo sau (lagging) VN-Index trong từng giai đoạn và tần số? *(trả lời bằng Wavelet-Granger)*
3. Mức độ liên kết có gia tăng trong các thời kỳ khủng hoảng so với giai đoạn bình thường? *(trả lời bằng Event Window Analysis)*
4. S&P 500 và SSE — thị trường nào có mức độ liên kết với VN-Index mạnh hơn?
5. **[MỚI]** Biến động của từng tài sản toàn cầu giải thích bao nhiêu % phương sai của VN-Index theo từng frequency band? *(trả lời bằng MODWT)*

### 1.3 Đối tượng & phạm vi

**Đối tượng:** Tỷ suất lợi nhuận hàng ngày của 7 chỉ số/tài sản: VNI, BTC, OIL, GOLD, SP500, SSE, VIX

**Phân kỳ giai đoạn:**

| Giai đoạn | Thời gian | Đặc trưng |
|---|---|---|
| Trước COVID-19 | 2015–2019 | Thị trường tương đối ổn định, BTC bùng nổ 2017, bong bóng SSE 2015 |
| COVID-19 | 2020–2021 | Cú sốc y tế toàn cầu, nới lỏng tiền tệ đại trà |
| Hậu COVID & Nga–Ukraine | 2022–2025 | Địa chính trị bất ổn, lạm phát cao, thắt chặt tiền tệ |

**Event windows (phân tích chuyên sâu):**

| Sự kiện | Cửa sổ phân tích | Lý do chọn |
|---|---|---|
| Bong bóng SSE 2015 | 2015-06-01 → 2015-09-30 | Kiểm định kênh lan truyền TQ → VN |
| BTC Crash tháng 5/2021 | 2021-04-01 → 2021-07-31 | BTC giảm ~50% trong 1 tháng |
| Fed tăng lãi suất mạnh 2022 | 2022-03-01 → 2022-12-31 | Tác động SP500 và kênh tài chính → VN |

---

## 3. Chương 2 — Tổng quan lý thuyết & nghiên cứu thực nghiệm

### 2.1 Cơ sở lý thuyết

**Lý thuyết 1 — Modern Portfolio Theory (Markowitz, 1952)**
- Nhà đầu tư tối ưu hóa danh mục bằng cách kết hợp tài sản có tương quan thấp hoặc âm
- Áp dụng: phân tích vai trò của BTC, Vàng, Dầu trong việc đa dạng hóa danh mục VN-Index

**Lý thuyết 2 — Safe Haven Hypothesis (Baur & Lucey, 2010)**
- Phân biệt: *hedge* (tương quan âm trung bình) vs *safe haven* (tương quan âm trong khủng hoảng)
- Áp dụng: kiểm định vai trò của Vàng, BTC như tài sản phòng ngừa/trú ẩn với VN-Index

**Lý thuyết 3 — Financial Contagion & Spillover**
- Lan truyền rủi ro giữa các thị trường tài chính — gia tăng trong khủng hoảng
- Kênh tài chính: SP500, BTC → VN (qua dòng vốn ngoại, tâm lý nhà đầu tư)
- Kênh thương mại: Oil, SSE → VN (qua giá hàng hóa, xuất nhập khẩu)

**Lý thuyết 4 — [MỚI] Uncertainty & Fear Transmission (VIX)**
- VIX (CBOE Volatility Index) đại diện cho sợ hãi thị trường toàn cầu — "Fear Index"
- Cơ chế: VIX tăng → flight to quality → vốn rút khỏi thị trường mới nổi → VNI giảm
- Tham khảo: Whaley (2000); Bekaert et al. (2013)

### 2.2 Sơ đồ tổng quan tài liệu tham khảo

```
Nhóm 1: Dầu–Vàng–Chứng khoán
├── GCC: Maghyereh et al. (2017) — DCC-GARCH
├── Toàn cầu: Basher & Sadorsky (2016) — DCC-GARCH
├── BRICS: Ansari & Sensarma (2019)
├── Châu Á (COVID): Prabheesh et al. (2020) — DCC-GARCH
├── Mỹ: Uddin et al. (2020); Bouri et al. (2021) — TVP-VAR
└── Việt Nam: Ngô Thái Hưng et al. (2022) — EGARCH

Nhóm 2: Bitcoin–Chứng khoán–Tài sản khác
├── Toàn cầu: Elsayed et al. (2022) — TVP-VAR
├── Hồi quy phân vị: Rao et al. (2022)
├── ĐNA: Kakinuma (2022); Lại Minh Khôi & Ngô Thái Hưng (2022)
├── Việt Nam: Ngo & Nguyen (2021) — DCC-GARCH
├── Việt Nam: Thanh et al. (2023) — TVP-VAR
└── Tài sản trú ẩn: Nguyen et al. (2024); Thuy et al. (2024)

Nhóm 3: Wavelet Coherence
├── Phương pháp: Torrence & Compo (1998); Torrence & Webster (1999)
├── Ứng dụng: Grinsted et al. (2004)
├── MODWT: Percival & Walden (2000) — [MỚI]
├── Wavelet-Granger: Aguiar-Conraria & Soares (2011) — [MỚI]
├── Dầu–Chứng khoán: Ali et al. (2022) — Wavelet + Granger
└── Bài báo gốc: Bùi Thị Thu Thảo & Cao Tấn Huy (2025)

Nhóm 4: S&P 500, SSE & VIX — Thị trường mới nổi [MỚI — cần tìm thêm]
├── Spillover SP500 → emerging markets Asia
├── SSE → Vietnam / ASEAN stock market
├── VIX → emerging markets (Bekaert et al., 2013)
└── US–China financial integration
```

### 2.3 Khoảng trống nghiên cứu

Khóa luận này lấp đầy khoảng trống so với Bùi Thị Thu Thảo & Cao Tấn Huy (2025) bằng cách:

1. **S&P 500** — phân tích kênh lan truyền từ thị trường tài chính phát triển lớn nhất thế giới
2. **SSE** — phân tích kênh lan truyền từ đối tác thương mại số 1 của Việt Nam
3. **VIX** — kiểm soát yếu tố "sợ hãi thị trường toàn cầu" như biến giải thích độc lập
4. **MODWT** — định lượng % phương sai VNI được giải thích theo từng frequency band
5. **Wavelet-Granger** — xác định chiều nhân quả, không chỉ mức độ đồng biến
6. **Event Window Analysis** — phân tích sâu 3 sự kiện cụ thể
7. **Giai đoạn 2015–2025** — bao phủ thêm hậu xung đột Nga–Ukraine đến 2025

---

## 4. Chương 3 — Dữ liệu & Phương pháp nghiên cứu

### 3.1 Dữ liệu nghiên cứu

**Bảng biến nghiên cứu:**

| Biến | Mô tả | Nguồn | Ticker | Ghi chú |
|---|---|---|---|---|
| VNI | Tỷ suất lợi nhuận VN-Index | vnstock (VCI API) | VNINDEX | Anchor cho date index |
| BTC | Tỷ suất lợi nhuận Bitcoin | Yahoo Finance | BTC-USD | |
| OIL | Tỷ suất lợi nhuận Dầu WTI | Yahoo Finance | CL=F | |
| GOLD | Tỷ suất lợi nhuận Vàng | Yahoo Finance | GC=F | |
| SP500 | Tỷ suất lợi nhuận S&P 500 | Yahoo Finance | ^GSPC | |
| SSE | Tỷ suất lợi nhuận Shanghai Composite | Yahoo Finance | 000001.SS | Xem ghi chú (1) |
| VIX | Chỉ số biến động CBOE | Yahoo Finance | ^VIX | [MỚI] |

> **(1) Ghi chú SSE:** Sử dụng `000001.SS` (SSE Composite) là biến chính. Chạy thêm robustness check với `000300.SS` (CSI 300) để kiểm tra tính vững của kết quả — CSI 300 đại diện tốt hơn cho blue-chip Trung Quốc và ít lỗi dữ liệu hơn.

**Công thức tỷ suất lợi nhuận (giống bài báo gốc):**

$$r_t = 100 \times \ln\left(\frac{P_t}{P_{t-1}}\right)$$

**Pipeline xử lý dữ liệu:**

```
Bước 1 — Thu thập
  VNIndex: vnstock VCI API
  Các chỉ số còn lại: yfinance.download(auto_adjust=True)
  Chuẩn hóa date → YYYY-MM-DD, loại bỏ timezone

Bước 2 — Date index (VNIndex-anchored)
  Chỉ giữ ngày VNIndex có giao dịch thực

Bước 3 — Merge & xử lý NaN
  OIL, GOLD, SP500, VIX nghỉ lễ Mỹ (≤3 ngày) → ffill(limit=3)
  SSE nghỉ Tết TQ (7–10 ngày)               → ffill(limit=3) + interpolate(limit=5)
  BTC thiếu đầu 2015 (~2 tuần)              → Không fill, ghi chú báo cáo chất lượng
  Còn lại sau fill                           → drop row, ghi nhận số lượng

Bước 4 — Tính log-return × 100
Bước 5 — Lưu file
```

**Chính sách xử lý NaN chi tiết:**

| Nguồn NaN | Nguyên nhân | Chiến lược |
|---|---|---|
| OIL, GOLD, SP500, VIX nghỉ lễ Mỹ (≤3 ngày) | Thị trường Mỹ đóng, VN mở | `ffill(limit=3)` |
| SSE nghỉ Tết TQ (7–10 ngày) | Thị trường TQ đóng dài | `ffill(limit=3)` + `interpolate(limit=5)` |
| BTC thiếu đầu 2015 (~2 tuần) | Dữ liệu không tồn tại | Không fill — ghi chú |
| Còn lại sau fill | Lỗi kỹ thuật hoặc quá dài | Drop hàng, ghi nhận |

**Code thu thập dữ liệu:**

```python
# 01_crawl_vnindex.py
from vnstock import Quote
import pandas as pd

vnindex = Quote(symbol='VNINDEX', source='VCI').history(
    start='2015-01-01', end='2025-12-31', interval='1D'
)
vnindex = vnindex[['time', 'close']].rename(columns={'time': 'date', 'close': 'close_vnindex'})
vnindex['date'] = pd.to_datetime(vnindex['date']).dt.date
vnindex.to_csv('data/raw/vnindex_raw.csv', index=False)

# 02_crawl_global.py
import yfinance as yf
import pandas as pd

TICKERS = {
    'btc'  : 'BTC-USD',
    'oil'  : 'CL=F',
    'gold' : 'GC=F',
    'sp500': '^GSPC',
    'sse'  : '000001.SS',
    'vix'  : '^VIX',       # [MỚI]
}

for name, ticker in TICKERS.items():
    df = yf.download(ticker, start='2015-01-01', end='2025-12-31', auto_adjust=True)
    df = df[['Close']].reset_index()
    df.columns = ['date', f'close_{name}']
    df['date'] = pd.to_datetime(df['date']).dt.date
    df.to_csv(f'data/raw/{name}_raw.csv', index=False)

# 03_merge_process.py
import numpy as np
import pandas as pd

vnindex = pd.read_csv('data/raw/vnindex_raw.csv', parse_dates=['date'])
global_dfs = {}
for name in ['btc', 'oil', 'gold', 'sp500', 'sse', 'vix']:
    global_dfs[name] = pd.read_csv(f'data/raw/{name}_raw.csv', parse_dates=['date'])

# Merge theo VNIndex-anchored date index
df = vnindex.copy()
for name, gdf in global_dfs.items():
    df = df.merge(gdf, on='date', how='left')

# Xử lý NaN
short_fill_cols = ['close_oil', 'close_gold', 'close_sp500', 'close_vix']
for col in short_fill_cols:
    df[col] = df[col].ffill(limit=3)

df['close_sse'] = df['close_sse'].ffill(limit=3)
df['close_sse'] = df['close_sse'].interpolate(method='linear', limit=5)

# Tính log-return
price_cols = ['close_vnindex', 'close_btc', 'close_oil', 'close_gold',
              'close_sp500', 'close_sse', 'close_vix']
return_cols = []
for col in price_cols:
    ret_col = col.replace('close_', 'return_')
    df[ret_col] = 100 * np.log(df[col] / df[col].shift(1))
    return_cols.append(ret_col)

df_returns = df[['date'] + return_cols].dropna()

# Lưu
df[['date'] + price_cols].to_csv('data/processed/master_price.csv', index=False)
df_returns.to_csv('data/processed/master_return.csv', index=False)

# Báo cáo chất lượng
quality = pd.DataFrame({
    'column': price_cols,
    'total_rows': [len(df)] * len(price_cols),
    'nan_before_fill': [df[c].isna().sum() for c in price_cols],
})
quality.to_csv('data/processed/data_quality_report.csv', index=False)
print("Pipeline hoàn tất.")
```

### 3.2 Phương pháp Wavelet Coherence (WTC)

**Lý do chọn Wavelet Coherence:**

| Phương pháp | Giới hạn |
|---|---|
| Tương quan Pearson | Chỉ phản ánh mối quan hệ trung bình toàn kỳ |
| GARCH / DCC-GARCH | Phân tích trong miền thời gian, không xác định được chu kỳ tần số |
| VAR / TVP-VAR | Không phân biệt mối quan hệ ngắn hạn và dài hạn theo tần số |
| **Wavelet Coherence** | **Phân tích đồng thời cả thời gian và tần số** |

**Công thức Wavelet Coherence bình phương:**

$$R^2(n,s) = \frac{|S(s^{-1}W^{XY}(n,s))|^2}{S(s^{-1}|W^X(n,s)|^2) \cdot S(s^{-1}|W^Y(n,s)|^2)}$$

Trong đó $R^2(n,s) \in [0,1]$ là hệ số đồng biến cục bộ tại thời điểm $n$, thang thời gian $s$.

**Góc pha — xác định mối quan hệ lead-lag:**

$$\phi_{xy}(n,s) = \tan^{-1}\left(\frac{\text{Im}(W^{XY}(n,s))}{\text{Re}(W^{XY}(n,s))}\right)$$

**Đọc biểu đồ WTC:**

| Mũi tên | Ý nghĩa |
|---|---|
| → (phải) | Đồng pha — tương quan dương |
| ← (trái) | Ngược pha — tương quan âm |
| ↗ (lên phải) | Đồng pha, X (tài sản) dẫn Y (VNI) |
| ↘ (xuống phải) | Đồng pha, Y (VNI) dẫn X (tài sản) |
| ↖ (lên trái) | Ngược pha, Y (VNI) dẫn X (tài sản) |
| ↙ (xuống trái) | Ngược pha, X (tài sản) dẫn Y (VNI) |

**Miền tần số / thang thời gian:**
- **Ngắn hạn:** scale 4–16 ngày (~1–3 tuần)
- **Trung hạn:** scale 16–64 ngày (~1–3 tháng)
- **Dài hạn:** scale 64–256 ngày (~3 tháng–1 năm)

**Code triển khai WTC:**

```python
# 06_wavelet_coherence.py
import pycwt as wavelet
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PAIRS = {
    'BTC_VNI'  : ('return_btc',   'return_vnindex'),
    'OIL_VNI'  : ('return_oil',   'return_vnindex'),
    'GOLD_VNI' : ('return_gold',  'return_vnindex'),
    'SP500_VNI': ('return_sp500', 'return_vnindex'),
    'SSE_VNI'  : ('return_sse',   'return_vnindex'),
    'VIX_VNI'  : ('return_vix',   'return_vnindex'),  # [MỚI]
}

PERIODS = {
    'pre_covid'  : ('2015-01-01', '2019-12-31'),
    'covid'      : ('2020-01-01', '2021-12-31'),
    'post_covid' : ('2022-01-01', '2025-12-31'),
}

def compute_wtc(x, y, dt=1, dj=1/12, s0=2, J=-1):
    mother = wavelet.Morlet(6)
    WCT, aWCT, coi, freq, sig = wavelet.wct(
        x, y, dt=dt, dj=dj, s0=s0, J=J,
        significance_level=0.95,
        wavelet=mother,
        normalize=True
    )
    periods = 1 / freq
    return WCT, aWCT, coi, periods, sig

def plot_wtc(WCT, aWCT, coi, periods, sig, time, title, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.RdBu_r
    levels = np.linspace(0, 1, 100)
    ax.contourf(time, np.log2(periods), WCT, levels=levels, cmap=cmap, extend='both')
    ax.contour(time, np.log2(periods), sig, [-99, 1], colors='k', linewidths=2)
    ax.fill(
        np.concatenate([time, time[-1:]+1, time[-1:]+1, time[:1]-1, time[:1]-1]),
        np.concatenate([np.log2(coi), [1e-9], np.log2([max(periods)]*2), np.log2([max(periods)]*2), [1e-9]]),
        'white', alpha=0.4, hatch='x'
    )
    # Vẽ mũi tên pha
    step_x = max(1, len(time) // 30)
    step_y = max(1, len(periods) // 20)
    for i in range(0, len(time), step_x):
        for j in range(0, len(periods), step_y):
            angle = aWCT[j, i]
            dx = np.cos(angle) * 0.4
            dy = np.sin(angle) * 0.4
            ax.annotate('', xy=(time[i]+dx, np.log2(periods[j])+dy),
                        xytext=(time[i], np.log2(periods[j])),
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Thời gian')
    ax.set_ylabel('Thang thời gian (ngày, log2)')
    yticks = [4, 8, 16, 32, 64, 128, 256]
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels([str(t) for t in yticks])
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0,1)),
                 ax=ax, label='R²')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

df = pd.read_csv('data/processed/master_return.csv', parse_dates=['date'])

for pair_name, (col_x, col_y) in PAIRS.items():
    for period_name, (start, end) in PERIODS.items():
        mask = (df['date'] >= start) & (df['date'] <= end)
        df_p = df[mask].reset_index(drop=True)
        x = df_p[col_x].values
        y = df_p[col_y].values
        WCT, aWCT, coi, periods, sig = compute_wtc(x, y)
        time = np.arange(len(x))
        title = f'Wavelet Coherence: {pair_name} — {period_name}'
        save_path = f'results/figures/wavelet_{pair_name.lower()}_{period_name}.png'
        plot_wtc(WCT, aWCT, coi, periods, sig, time, title, save_path)
        print(f'✓ {pair_name} / {period_name}')
```

### 3.3 [MỚI] MODWT Variance Decomposition

**Mục đích:** Phân rã phương sai lợi nhuận VNI theo từng frequency band để định lượng "tỷ trọng đóng góp" của mỗi tần số vào tổng biến động.

**Lý thuyết cơ bản (Percival & Walden, 2000):**

$$\text{Var}(r_t) = \sum_{j=1}^{J} \sigma^2_j + \sigma^2_J$$

Trong đó $\sigma^2_j$ là phương sai tại scale $j$ (tính từ hệ số MODWT).

**Bổ sung:** So sánh wavelet variance của VNI với từng tài sản toàn cầu → xác định tài sản nào có cấu trúc tần số tương đồng nhất với VNI.

```python
# 07_modwt_variance.py
import pywt
import numpy as np
import pandas as pd

def modwt_variance(x, wavelet='db4', level=6):
    """Tính MODWT variance decomposition."""
    coeffs = pywt.swt(x, wavelet, level=level, norm=True)
    variances = {}
    for j, (cA, cD) in enumerate(reversed(coeffs), start=1):
        scale_days = 2**j
        variances[f'scale_{scale_days}d'] = np.var(cD)
    variances['smooth'] = np.var(coeffs[0][0])
    total = sum(variances.values())
    proportions = {k: v/total*100 for k, v in variances.items()}
    return variances, proportions

df = pd.read_csv('data/processed/master_return.csv', parse_dates=['date'])
return_cols = ['return_vnindex', 'return_btc', 'return_oil', 'return_gold',
               'return_sp500', 'return_sse', 'return_vix']

results = {}
for col in return_cols:
    x = df[col].dropna().values
    variances, proportions = modwt_variance(x)
    results[col] = proportions

pd.DataFrame(results).T.to_csv('results/tables/modwt_variance.csv')
print("MODWT Variance Decomposition hoàn tất.")
```

**Bảng kết quả cần tạo:**

| Biến | Scale 4d (%) | Scale 8d (%) | Scale 16d (%) | Scale 32d (%) | Scale 64d (%) | Scale 128d (%) | Smooth (%) |
|---|---|---|---|---|---|---|---|
| VNI | | | | | | | |
| BTC | | | | | | | |
| OIL | | | | | | | |
| GOLD | | | | | | | |
| SP500 | | | | | | | |
| SSE | | | | | | | |
| VIX | | | | | | | |

*(Điền sau khi chạy phân tích)*

### 3.4 [MỚI] Wavelet-Granger Causality

**Mục đích:** Giải quyết hạn chế của WTC — xác định chiều nhân quả, không chỉ mức độ đồng biến.

**Cơ sở lý thuyết (Aguiar-Conraria & Soares, 2011):**
Phân rã chuỗi theo MODWT → kiểm định Granger causality tại từng frequency band → xác định tại thang nào X (tài sản toàn cầu) gây ra Y (VNI).

```python
# 08_wavelet_granger.py
import pywt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

def wavelet_granger(x, y, wavelet='db4', level=6, max_lag=5):
    """
    Kiểm định Granger causality tại từng frequency band.
    Trả về p-value tại mỗi scale.
    """
    coeffs_x = pywt.swt(x, wavelet, level=level, norm=True)
    coeffs_y = pywt.swt(y, wavelet, level=level, norm=True)

    results = {}
    for j in range(level):
        scale_days = 2**(j+1)
        dx = coeffs_x[level-1-j][1]  # detail coefficients tại scale j
        dy = coeffs_y[level-1-j][1]

        # Granger test: X → Y tại scale này
        data_xy = np.column_stack([dy, dx])
        data_yx = np.column_stack([dx, dy])

        gc_xy = grangercausalitytests(data_xy, maxlag=max_lag, verbose=False)
        gc_yx = grangercausalitytests(data_yx, maxlag=max_lag, verbose=False)

        p_xy = min([gc_xy[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)])
        p_yx = min([gc_yx[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)])

        results[f'scale_{scale_days}d'] = {
            'p_X_causes_Y': round(p_xy, 4),  # tài sản → VNI
            'p_Y_causes_X': round(p_yx, 4),  # VNI → tài sản
            'direction': 'X→Y' if p_xy < 0.05 else ('Y→X' if p_yx < 0.05 else 'None')
        }
    return results

df = pd.read_csv('data/processed/master_return.csv', parse_dates=['date'])

PAIRS = {
    'BTC_VNI'  : ('return_btc',   'return_vnindex'),
    'OIL_VNI'  : ('return_oil',   'return_vnindex'),
    'GOLD_VNI' : ('return_gold',  'return_vnindex'),
    'SP500_VNI': ('return_sp500', 'return_vnindex'),
    'SSE_VNI'  : ('return_sse',   'return_vnindex'),
    'VIX_VNI'  : ('return_vix',   'return_vnindex'),
}

PERIODS = {
    'pre_covid'  : ('2015-01-01', '2019-12-31'),
    'covid'      : ('2020-01-01', '2021-12-31'),
    'post_covid' : ('2022-01-01', '2025-12-31'),
}

all_results = []
for pair_name, (col_x, col_y) in PAIRS.items():
    for period_name, (start, end) in PERIODS.items():
        mask = (df['date'] >= start) & (df['date'] <= end)
        df_p = df[mask].reset_index(drop=True)
        x = df_p[col_x].values
        y = df_p[col_y].values
        res = wavelet_granger(x, y)
        for scale, vals in res.items():
            all_results.append({
                'pair': pair_name, 'period': period_name, 'scale': scale, **vals
            })

pd.DataFrame(all_results).to_csv('results/tables/wavelet_granger.csv', index=False)
print("Wavelet-Granger Causality hoàn tất.")
```

**Bảng kết quả Wavelet-Granger (template):**

| Cặp | Scale | Trước COVID | Trong COVID | Hậu COVID |
|---|---|---|---|---|
| SP500 → VNI | 4d | | | |
| SP500 → VNI | 16d | | | |
| SP500 → VNI | 64d | | | |
| SSE → VNI | 4d | | | |
| ... | ... | | | |

*(Điền: ✓ p<0.05 / ✗ p≥0.05)*

### 3.5 [MỚI] Event Window Analysis

**Mục đích:** Phân tích chuyên sâu mức độ liên kết trong 3 sự kiện cụ thể — bằng chứng về "contagion effect".

**3 event windows:**

| Sự kiện | Cửa sổ phân tích | Biến kỳ vọng phản ứng mạnh |
|---|---|---|
| Bong bóng SSE 2015 | 2015-06-01 → 2015-09-30 | SSE, VNI |
| BTC Crash tháng 5/2021 | 2021-04-01 → 2021-07-31 | BTC, VIX, VNI |
| Fed tăng lãi suất mạnh 2022 | 2022-03-01 → 2022-12-31 | SP500, VIX, GOLD, VNI |

**Phân tích cho mỗi event window:**
1. Vẽ biểu đồ WTC riêng (zoom vào giai đoạn)
2. Tính average coherence (R²) trước / trong / sau sự kiện
3. So sánh với coherence trung bình toàn kỳ → đo "contagion premium"

```python
# 09_event_window.py
import pandas as pd
import numpy as np
import pycwt as wavelet

EVENT_WINDOWS = {
    'sse_bubble_2015' : {
        'pre'   : ('2015-01-01', '2015-05-31'),
        'event' : ('2015-06-01', '2015-09-30'),
        'post'  : ('2015-10-01', '2015-12-31'),
        'pairs' : ['SSE_VNI', 'SP500_VNI'],
    },
    'btc_crash_2021' : {
        'pre'   : ('2021-01-01', '2021-03-31'),
        'event' : ('2021-04-01', '2021-07-31'),
        'post'  : ('2021-08-01', '2021-12-31'),
        'pairs' : ['BTC_VNI', 'VIX_VNI'],
    },
    'fed_hike_2022' : {
        'pre'   : ('2022-01-01', '2022-02-28'),
        'event' : ('2022-03-01', '2022-12-31'),
        'post'  : ('2023-01-01', '2023-06-30'),
        'pairs' : ['SP500_VNI', 'VIX_VNI', 'GOLD_VNI'],
    },
}

# Tính average coherence cho mỗi window và so sánh
# → output: bảng avg_coherence_event_analysis.csv
```

---

## 5. Chương 4 — Kết quả & Thảo luận

### 4.1 Thống kê mô tả

**Bảng cần tạo:**

| Thống kê | VNI | BTC | OIL | GOLD | SP500 | SSE | VIX |
|---|---|---|---|---|---|---|---|
| Số quan sát | | | | | | | |
| Trung bình (%) | | | | | | | |
| Độ lệch chuẩn (%) | | | | | | | |
| Min (%) | | | | | | | |
| Max (%) | | | | | | | |
| Skewness | | | | | | | |
| Kurtosis | | | | | | | |
| Jarque-Bera (p-value) | | | | | | | |

```python
# 04_descriptive_stats.py
from scipy import stats
import pandas as pd

df = pd.read_csv('data/processed/master_return.csv', parse_dates=['date'])
return_cols = ['return_vnindex', 'return_btc', 'return_oil', 'return_gold',
               'return_sp500', 'return_sse', 'return_vix']

desc = {}
for col in return_cols:
    s = df[col].dropna()
    jb_stat, jb_p = stats.jarque_bera(s)
    desc[col] = {
        'N': len(s), 'Mean': round(s.mean(), 4), 'Std': round(s.std(), 4),
        'Min': round(s.min(), 4), 'Max': round(s.max(), 4),
        'Skewness': round(s.skew(), 4), 'Kurtosis': round(s.kurtosis(), 4),
        'JB_stat': round(jb_stat, 4), 'JB_p': round(jb_p, 4),
    }
pd.DataFrame(desc).T.to_csv('results/tables/descriptive_stats.csv')
```

### 4.2 Ma trận tương quan Pearson

```python
# 05_correlation.py
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

df = pd.read_csv('data/processed/master_return.csv')
return_cols = ['return_vnindex', 'return_btc', 'return_oil',
               'return_gold', 'return_sp500', 'return_sse', 'return_vix']

corr = df[return_cols].corr()
# Tính p-value cho từng cặp
pval = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns)
for i in corr.index:
    for j in corr.columns:
        if i != j:
            _, p = pearsonr(df[i].dropna(), df[j].dropna())
            pval.loc[i, j] = round(p, 4)

corr.to_csv('results/tables/correlation_matrix.csv')
pval.to_csv('results/tables/correlation_pvalue.csv')
```

**Kỳ vọng phát hiện:**

| Cặp | Kỳ vọng | Lý giải |
|---|---|---|
| VNI – OIL | Dương nhẹ | Kinh tế VN phụ thuộc năng lượng |
| VNI – GOLD | Âm nhẹ | Vàng là tài sản trú ẩn |
| VNI – BTC | ~0 | Độc lập ngắn hạn |
| VNI – SP500 | Dương | Hội nhập tài chính toàn cầu |
| VNI – SSE | Dương | Quan hệ thương mại VN–TQ |
| VNI – VIX | **Âm** | VIX tăng → vốn rút khỏi EM → VNI giảm |

### 4.3 Kiểm định tính dừng (ADF)

```python
from statsmodels.tsa.stattools import adfuller

adf_results = []
for col in return_cols:
    result = adfuller(df[col].dropna(), autolag='AIC')
    adf_results.append({
        'Variable': col,
        'ADF Statistic': round(result[0], 4),
        'p-value': round(result[1], 4),
        'Critical 1%': round(result[4]['1%'], 4),
        'Critical 5%': round(result[4]['5%'], 4),
        'Kết luận': 'Dừng I(0)' if result[1] < 0.05 else 'Không dừng',
    })
pd.DataFrame(adf_results).to_csv('results/tables/adf_test.csv', index=False)
```

**Kỳ vọng:** Tất cả chuỗi log-return đều dừng ở I(0) — p-value < 0.01

### 4.4 [MỚI] MODWT — Phân rã phương sai theo frequency band

Trình bày bảng MODWT variance decomposition (xem 3.3). Thảo luận:
- Tần số nào đóng góp nhiều nhất vào tổng biến động VNI?
- Cấu trúc tần số của VNI tương đồng nhất với tài sản nào?
- Hàm ý: nhà đầu tư ở horizon nào chịu rủi ro lan truyền nhiều nhất?

### 4.5 Kết quả Wavelet Coherence — theo từng giai đoạn

**18 biểu đồ WTC** (6 cặp × 3 giai đoạn):

```
results/figures/
├── wavelet_btc_vni_pre_covid.png
├── wavelet_btc_vni_covid.png
├── wavelet_btc_vni_post_covid.png
├── wavelet_oil_vni_pre_covid.png
├── ... (18 biểu đồ tổng cộng)
└── wavelet_vix_vni_post_covid.png
```

**Template thảo luận cho cặp SP500–VNI (mới):**

```
Trước COVID (2015-2019):
- Liên kết ngắt quãng ở thang ngắn, chủ yếu ở thang trung-dài
- SP500 dẫn VNI (↗) ở thang 32-64 ngày
- Phản ánh: tác động gián tiếp qua kênh tâm lý và dòng vốn ngoại

Trong COVID (2020-2021):
- Liên kết tăng mạnh, đặc biệt Q1/2020 (crash toàn thị trường)
- SP500 dẫn VNI rõ ràng → cú sốc phát sinh từ thị trường Mỹ
- Giai đoạn phục hồi 2021: liên kết trung hạn, đồng pha → nới lỏng tiền tệ đồng bộ

Hậu COVID & Nga-Ukraine (2022-2025):
- Liên kết cao và nhất quán hơn → tăng hội nhập tài chính
- Fed tăng lãi suất 2022: SP500 giảm → VNI giảm, SP500 dẫn rõ ràng
- Hàm ý: VN-Index ngày càng nhạy cảm với chính sách tiền tệ Mỹ
```

**Template thảo luận cho cặp SSE–VNI (mới):**

```
Trước COVID (2015-2019):
- Bong bóng SSE 2015: liên kết tăng mạnh giai đoạn 6-9/2015
- SSE dẫn VNI ở thang ngắn → cú sốc lan truyền qua kênh thương mại
- Ngoài giai đoạn này: liên kết yếu → quan hệ gián tiếp

Trong COVID (2020-2021):
- TQ phát sinh dịch trước → SSE phản ứng trước VNI
- Thang ngắn-trung: SSE dẫn VNI (↗) rõ ràng Q1/2020
- Phục hồi 2021: liên kết trung hạn, đồng pha

Hậu COVID & Nga-Ukraine (2022-2025):
- Liên kết ổn định ở thang trung hạn
- So sánh với SP500: SSE có thể yếu hơn ở thang ngắn (tài chính),
  nhưng mạnh hơn ở thang dài (thương mại)
```

### 4.6 [MỚI] Wavelet-Granger Causality — theo frequency band

Trình bày bảng tổng hợp kết quả Granger causality tại từng scale × giai đoạn × cặp. Thảo luận:
- Tại thang nào SP500 dẫn VNI mạnh nhất?
- SSE có dẫn VNI ở thang dài hơn SP500 không (kênh thương mại vs tài chính)?
- VIX có quan hệ nhân quả một chiều với VNI không?

### 4.7 [MỚI] Event Window Analysis — 3 sự kiện

**Bảng so sánh average coherence (R²) theo sự kiện:**

| Sự kiện | Cặp | Pre-event | During event | Post-event | Contagion premium |
|---|---|---|---|---|---|
| SSE Bubble 2015 | SSE–VNI | | | | |
| BTC Crash 2021 | BTC–VNI | | | | |
| BTC Crash 2021 | VIX–VNI | | | | |
| Fed Hike 2022 | SP500–VNI | | | | |
| Fed Hike 2022 | VIX–VNI | | | | |
| Fed Hike 2022 | GOLD–VNI | | | | |

*Contagion premium = R²(during) − R²(pre)*

### 4.8 [MỚI] Average Coherence Heatmap — tổng hợp toàn bộ

Tạo heatmap (6 cặp × 3 giai đoạn × 3 frequency bands) thể hiện average R² — dễ đọc hơn 18 biểu đồ WTC.

```python
# 10_coherence_heatmap.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pycwt as wavelet

FREQUENCY_BANDS = {
    'short'  : (4,  16),   # ngày
    'medium' : (16, 64),
    'long'   : (64, 256),
}

def avg_coherence_by_band(WCT, periods, band):
    """Tính R² trung bình trong một frequency band."""
    low, high = band
    mask = (periods >= low) & (periods < high)
    return float(np.nanmean(WCT[mask, :]))

# Tính cho tất cả cặp × giai đoạn × bands
# → lưu vào results/tables/avg_coherence_heatmap.csv
# → vẽ heatmap với seaborn

# Ví dụ vẽ heatmap
data = pd.read_csv('results/tables/avg_coherence_heatmap.csv', index_col=0)
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(data, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=0, vmax=0.6, linewidths=0.5, ax=ax)
ax.set_title('Average Wavelet Coherence (R²) — theo cặp, giai đoạn & tần số', fontsize=13)
plt.tight_layout()
plt.savefig('results/figures/coherence_heatmap_summary.png', dpi=150)
```

### 4.9 Bảng tổng hợp kết quả

**Bảng X. Kết quả Wavelet Coherence, Granger Causality & Event Analysis:**

| Cặp biến | Tần số | Trước COVID | Trong COVID | Hậu COVID | Lead/Lag chủ đạo |
|---|---|---|---|---|---|
| BTC–VNI | Ngắn | | | | |
| BTC–VNI | Trung | | | | |
| BTC–VNI | Dài | | | | |
| OIL–VNI | Ngắn | | | | |
| ... | ... | | | | |
| VIX–VNI | Ngắn | | | | |
| VIX–VNI | Trung | | | | |
| VIX–VNI | Dài | | | | |

*(Điền sau khi chạy phân tích)*

---

## 6. Chương 5 — Kết luận & Khuyến nghị

### 5.1 Cấu trúc kết luận

Trả lời 5 câu hỏi nghiên cứu (mục 1.2):

- **Bitcoin:** vai trò thay thế/đầu cơ → tăng liên kết trong khủng hoảng (đặc biệt COVID)
- **Dầu thô:** chỉ báo chu kỳ kinh tế → dẫn dắt VNI ở thang dài
- **Vàng:** tài sản trú ẩn → ngược pha trong khủng hoảng, liên kết âm với VNI
- **S&P 500:** kênh tài chính chủ đạo → mức độ hội nhập tăng theo thời gian; SP500 dẫn VNI nhất quán
- **SSE:** kênh thương mại → liên kết mạnh hơn ở thang dài; dẫn VNI rõ trong các sự kiện TQ-specific
- **VIX:** [MỚI] tương quan âm với VNI → xác nhận cơ chế "flight to quality" ảnh hưởng đến thị trường VN
- **MODWT:** [MỚI] định lượng % phương sai VNI theo tần số → biến động ngắn hạn chiếm X%, dài hạn chiếm Y%

### 5.2 Khuyến nghị

**Cho nhà đầu tư:**
- **Giai đoạn bình thường:** đa dạng hóa bằng Vàng và tài sản ít tương quan với VNI
- **Giai đoạn khủng hoảng:** Vàng là tài sản trú ẩn hiệu quả (liên kết ngược pha)
- **Chỉ báo sớm:** Theo dõi SP500 (thang ngắn–trung) và VIX như leading indicator cho VNI
- **Nhà đầu tư dài hạn:** quan tâm SSE như proxy cho rủi ro thương mại VN–TQ

**Cho cơ quan quản lý:**
- Giám sát kênh lan truyền từ SP500 (tài chính) và SSE (thương mại) đến VNI
- Chú ý VIX như cảnh báo sớm về áp lực rút vốn ngoại
- Giai đoạn bất ổn địa chính trị — mức độ liên kết tăng đột biến ở nhiều tần số

### 5.3 Hạn chế & hướng tiếp theo

**Hạn chế:**
- Chỉ phân tích tỷ suất lợi nhuận giá đóng cửa — chưa xét volume, volatility
- Wavelet Coherence xác định mối quan hệ đồng biến nhưng chưa khẳng định hoàn toàn nhân quả
- Chưa kiểm soát biến vĩ mô đồng thời (tỷ giá VND/USD, lãi suất FED trực tiếp, lạm phát)
- Dữ liệu SSE có giới hạn về volume (Yahoo Finance không cung cấp chính xác)

**Hướng nghiên cứu tiếp theo:**
- Mở rộng thêm biến vĩ mô (lãi suất FED, tỷ giá VND/USD, CPI)
- Xây dựng Connectedness Table (Diebold & Yılmaz, 2012) để đo tổng lan truyền rủi ro
- Phân tích riêng cho từng ngành trong VNI (tài chính, năng lượng, bất động sản)
- Mở rộng sang các thị trường ASEAN-6 để so sánh mức độ hội nhập

---

## 7. Pipeline dữ liệu chi tiết

### 7.1 Cấu trúc thư mục dự án

```
thesis_project/
│
├── data/
│   ├── raw/
│   │   ├── vnindex_raw.csv
│   │   ├── bitcoin_raw.csv
│   │   ├── oil_raw.csv
│   │   ├── gold_raw.csv
│   │   ├── sp500_raw.csv
│   │   ├── sse_raw.csv
│   │   └── vix_raw.csv                  # [MỚI]
│   │
│   └── processed/
│       ├── master_price.csv
│       ├── master_return.csv
│       └── data_quality_report.csv
│
├── scripts/
│   ├── 01_crawl_vnindex.py
│   ├── 02_crawl_global.py
│   ├── 03_merge_process.py
│   ├── 04_descriptive_stats.py
│   ├── 05_correlation.py
│   ├── 06_wavelet_coherence.py          # WTC — 18 biểu đồ
│   ├── 07_modwt_variance.py             # [MỚI] MODWT decomposition
│   ├── 08_wavelet_granger.py            # [MỚI] Wavelet-Granger causality
│   ├── 09_event_window.py               # [MỚI] Event window analysis
│   └── 10_coherence_heatmap.py          # [MỚI] Summary heatmap
│
├── results/
│   ├── tables/
│   │   ├── descriptive_stats.csv
│   │   ├── correlation_matrix.csv
│   │   ├── correlation_pvalue.csv
│   │   ├── adf_test.csv
│   │   ├── modwt_variance.csv           # [MỚI]
│   │   ├── wavelet_granger.csv          # [MỚI]
│   │   ├── event_window_coherence.csv   # [MỚI]
│   │   ├── avg_coherence_heatmap.csv    # [MỚI]
│   │   └── wavelet_summary_table.csv
│   │
│   └── figures/
│       ├── wavelet_btc_vni_*.png        # 3 biểu đồ
│       ├── wavelet_oil_vni_*.png        # 3 biểu đồ
│       ├── wavelet_gold_vni_*.png       # 3 biểu đồ
│       ├── wavelet_sp500_vni_*.png      # 3 biểu đồ [MỚI]
│       ├── wavelet_sse_vni_*.png        # 3 biểu đồ [MỚI]
│       ├── wavelet_vix_vni_*.png        # 3 biểu đồ [MỚI]
│       ├── event_sse_bubble_2015.png    # [MỚI]
│       ├── event_btc_crash_2021.png     # [MỚI]
│       ├── event_fed_hike_2022.png      # [MỚI]
│       └── coherence_heatmap_summary.png # [MỚI]
│
└── thesis/
    ├── thesis_draft.docx
    └── references.bib
```

### 7.2 Phân kỳ dữ liệu

```python
PERIODS = {
    'pre_covid'  : ('2015-01-01', '2019-12-31'),
    'covid'      : ('2020-01-01', '2021-12-31'),
    'post_covid' : ('2022-01-01', '2025-12-31'),
}

EVENT_WINDOWS = {
    'sse_bubble_2015' : ('2015-06-01', '2015-09-30'),
    'btc_crash_2021'  : ('2021-04-01', '2021-07-31'),
    'fed_hike_2022'   : ('2022-03-01', '2022-12-31'),
}
```

### 7.3 Robustness Check — SSE vs CSI 300

```python
# Chạy lại toàn bộ WTC và Granger với CSI 300 thay SSE
# So sánh kết quả để kiểm định tính vững
csi300 = yf.download('000300.SS', start='2015-01-01', end='2025-12-31', auto_adjust=True)
# Nếu kết quả nhất quán → kết luận vững
```

---

## 8. Checklist thực hiện

### Môi trường & cài đặt

- [ ] Tạo virtual environment: `python -m venv thesis_env`
- [ ] Cài packages: `pip install vnstock yfinance pycwt pywavelets statsmodels scipy seaborn matplotlib pandas numpy`
- [ ] Kiểm tra pycwt hoạt động đúng với `wavelet.wct()`
- [ ] Tạo cấu trúc thư mục theo section 7.1

### Thu thập & xử lý dữ liệu

- [ ] Crawl VNIndex (2015–2025) từ vnstock VCI
- [ ] Crawl BTC, OIL, GOLD, SP500, SSE, **VIX** từ yfinance
- [ ] Kiểm tra tỉ lệ NaN từng chỉ số — ghi vào báo cáo chất lượng
- [ ] Merge theo VNIndex-anchored date index
- [ ] Xử lý NaN theo policy đã định
- [ ] Tính log-return × 100
- [ ] Lưu `master_return.csv`
- [ ] Crawl CSI 300 (`000300.SS`) cho robustness check

### Phân tích thống kê cơ bản

- [ ] Thống kê mô tả 7 biến (N, Mean, Std, Min, Max, Skewness, Kurtosis, JB-test)
- [ ] Ma trận tương quan Pearson 7×7 + p-value
- [ ] Kiểm định ADF cho tất cả 7 chuỗi return
- [ ] Format bảng cho khóa luận

### MODWT Variance Decomposition [MỚI]

- [ ] Tính MODWT variance decomposition cho 7 biến (toàn kỳ)
- [ ] Tính theo 3 giai đoạn để so sánh
- [ ] Tạo bảng tổng hợp % variance theo frequency band

### Wavelet Coherence — 18 biểu đồ

- [ ] Tính WTC cho 6 cặp × 3 giai đoạn = 18 lần chạy
- [ ] Vẽ 18 biểu đồ WTC (màu sắc, mũi tên pha, COI, đường viền 5%)
- [ ] Điền bảng tổng hợp average coherence

### Wavelet-Granger Causality [MỚI]

- [ ] Chạy Granger test tại từng scale cho 6 cặp × 3 giai đoạn
- [ ] Tạo bảng tổng hợp chiều nhân quả theo scale và giai đoạn
- [ ] Viết thảo luận so sánh WTC vs Granger

### Event Window Analysis [MỚI]

- [ ] Event 1: SSE Bubble 2015 — WTC zoom + avg coherence table
- [ ] Event 2: BTC Crash tháng 5/2021 — WTC zoom + avg coherence table
- [ ] Event 3: Fed Hike 2022 — WTC zoom + avg coherence table
- [ ] Tính contagion premium cho từng sự kiện

### Average Coherence Heatmap [MỚI]

- [ ] Tính avg R² cho 6 cặp × 3 giai đoạn × 3 frequency bands
- [ ] Vẽ heatmap tổng hợp
- [ ] Export bảng số kèm heatmap

### Robustness Check

- [ ] Chạy lại SSE analysis với CSI 300 → so sánh kết quả

### Viết khóa luận

- [ ] Chương 1: Giới thiệu (~3–4 trang)
- [ ] Chương 2: Tổng quan lý thuyết (~8–10 trang) — tìm thêm tài liệu SP500/SSE/VIX
- [ ] Chương 3: Dữ liệu & Phương pháp (~6–8 trang) — mô tả 4 phương pháp
- [ ] Chương 4: Kết quả & Thảo luận (~15–20 trang + 18 biểu đồ + heatmap)
- [ ] Chương 5: Kết luận & Khuyến nghị (~3–4 trang)
- [ ] Phụ lục: Thống kê mô tả chi tiết, Ma trận tương quan, ADF, MODWT table

---

## 9. Tài liệu tham khảo

### Phương pháp Wavelet (BẮT BUỘC ĐỌC)

- **Torrence & Compo (1998)** — A practical guide to wavelet analysis — *Bull. Amer. Meteor. Soc.*
- **Torrence & Webster (1999)** — Mở rộng góc pha
- **Grinsted et al. (2004)** — Ứng dụng trong chuỗi thời gian địa vật lý
- **Percival & Walden (2000)** — Wavelet Methods for Time Series Analysis — **[MỚI — cho MODWT]**
- **Aguiar-Conraria & Soares (2011)** — The continuous wavelet transform: moving beyond uni- and bivariate analysis — **[MỚI — cho Wavelet-Granger]**

### Bài báo gốc & nền tảng trực tiếp

- **Bùi Thị Thu Thảo & Cao Tấn Huy (2025)** — Journal of Finance-Marketing Research — ĐỌC KỸ
- **Markowitz (1952)** — Portfolio Selection — *Journal of Finance*
- **Baur & Lucey (2010)** — Is gold a hedge or a safe haven? — *Financial Review*
- **Whaley (2000)** — The investor fear gauge — *Journal of Portfolio Management* — **[MỚI — cho VIX]**
- **Bekaert et al. (2013)** — VIX, the variance premium and stock market volatility — **[MỚI — cho VIX]**

### Nghiên cứu tại Việt Nam

- **Ngô Thái Hưng et al. (2022)** — EGARCH — Trái phiếu, cổ phiếu, dầu, vàng VN
- **Thanh et al. (2023)** — TVP-VAR — Tiền điện tử & TTCK VN
- **Lại Minh Khôi & Ngô Thái Hưng (2022)** — QQR — Bitcoin & ASEAN-6
- **Ngo & Nguyen (2021)** — DCC-GARCH — Bitcoin & VN

### Cần tìm thêm (ưu tiên)

- Spillover từ SP500 đến thị trường mới nổi châu Á (post-2020)
- Mối quan hệ SSE và thị trường ASEAN (wavelet hoặc DCC-GARCH)
- VIX và thị trường mới nổi: cơ chế "flight to quality"
- VN–Trung Quốc: kết nối tài chính và thương mại (2015–2025)

---

*Cập nhật: 2026 · Dựa trên Bùi Thị Thu Thảo & Cao Tấn Huy (2025), Journal of Finance-Marketing Research*
*Phiên bản 2.0 — Bổ sung VIX, MODWT, Wavelet-Granger, Event Window Analysis*