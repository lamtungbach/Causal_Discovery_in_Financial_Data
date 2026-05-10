# Khóa Luận Tốt Nghiệp — NOTEARS Causal Discovery

**Đề tài:** Phân tích Lan tỏa Rủi ro giữa các Thị trường Tài chính  
**Thuật toán chính:** NOTEARS  
**So sánh:** PC Algorithm, Granger Causality  

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc thư mục

```
thesis-notears/
├── config.py                  ← Tham số cấu hình (sửa ở đây)
├── run_data_pipeline.py       ← Chạy thu thập dữ liệu
├── run_models.py              ← Chạy 3 mô hình (sắp có)
├── requirements.txt
├── data/
│   ├── raw/                   ← Dữ liệu thô từ Yahoo Finance
│   ├── processed/             ← Log-return, volatility (INPUT NOTEARS)
│   └── results/               ← Kết quả mô hình, bảng so sánh
├── src/
│   ├── data/
│   │   ├── downloader.py      ← Tải dữ liệu Yahoo Finance
│   │   ├── processor.py       ← Tính log-return, volatility
│   │   └── validator.py       ← Kiểm định ADF/KPSS, thống kê mô tả
│   ├── models/
│   │   ├── granger.py         ← Baseline Granger Causality (sắp có)
│   │   ├── pc_algorithm.py    ← PC Algorithm (sắp có)
│   │   └── notears.py         ← NOTEARS chính (sắp có)
│   ├── evaluation/
│   │   └── metrics.py         ← SHD, Precision, Recall, F1 (sắp có)
│   └── visualization/
│       └── plotter.py         ← Biểu đồ volatility, DAG, heatmap
├── notebooks/
│   └── exploration.ipynb      ← Jupyter notebook khám phá dữ liệu
├── figures/                   ← Biểu đồ xuất ra (dùng trong luận văn)
└── reports/                   ← Bảng kết quả xuất ra

```

## Chạy từng bước

```bash
# Bước 1: Thu thập & xử lý dữ liệu
python run_data_pipeline.py

# Bước 2: Chạy 3 mô hình (sắp có)
python run_models.py

# Hoặc chạy từng module riêng
python src/data/downloader.py      # chỉ download
python src/data/processor.py       # chỉ xử lý
python src/data/validator.py       # chỉ kiểm định
python src/visualization/plotter.py # chỉ vẽ biểu đồ
```

## Tham số quan trọng (config.py)

| Tham số          | Mặc định | Ý nghĩa                        |
|-----------------|----------|-------------------------------|
| ROLLING_WINDOW  | 21       | Cửa sổ tính volatility (ngày) |
| NOTEARS_THRESH  | 0.2      | Ngưỡng cắt cạnh DAG           |
| NOTEARS_LAMBDA  | 0.1      | Regularization NOTEARS        |
| GRANGER_MAX_LAG | 5        | Lag tối đa Granger Causality  |
