# KSE 2026 Camera-ready Financial Causal Discovery

Dự án này là mã nguồn thực nghiệm cho bản sửa camera-ready của bài báo KSE 2026. Pipeline chính đã được điều chỉnh theo phản biện: xử lý lịch giao dịch và WTI minh bạch, chia dữ liệu theo ngày dự báo, chọn checkpoint bằng validation, so sánh VAR(20), kiểm định dự báo, đánh giá độ ổn định qua 20 seed và kiểm tra robustness.

> **Nguồn kết quả chính thức:** `scripts/14_camera_ready_revision.py` và `results/camera_ready/`. Các script `00–12` cùng `results/tables/`, `results/figures/` là kết quả cũ, chỉ giữ để đối chiếu và không nên dùng để cập nhật paper.

## Chạy nhanh

Yêu cầu Python 3.10+ và các thư viện trong `requirements.txt`.

```bash
pip install -r requirements.txt
```

Để tái tạo đúng môi trường đã kiểm tra, ưu tiên tạo virtual environment riêng và dùng `requirements-tested.txt`.

Kiểm tra nhanh luồng xử lý:

```bash
python run_pipeline.py --profile smoke --device auto --stability-seeds 2
```

Chạy thí nghiệm camera-ready đầy đủ và tự động kiểm tra đầu ra:

```bash
python run_pipeline.py --device auto
```

Nếu chắc chắn máy có CUDA:

```bash
python run_pipeline.py --device cuda
```

Chỉ kiểm tra lại bộ kết quả đầy đủ đã có:

```bash
python scripts/15_verify_camera_ready_outputs.py
```

Báo cáo rà soát gần nhất được lưu tại `PROJECT_AUDIT.md`.

Kiểm tra tái lập (huấn luyện seed 42 hai lần và đối chiếu seed 43):

```bash
python scripts/16_check_reproducibility.py --device auto
```

Mở dashboard camera-ready:

```bash
streamlit run app/13_dashboard_streamlit.py
```

## Hợp đồng thực nghiệm

- Biến: `VNI`, `SSE`, `SP500`, `OIL`, `GOLD`, `BTC`, `VIX`.
- Độ trễ: 20 phiên; NGC dùng MLP `140-64-64-1`; LSTM có hidden size 64.
- Chia theo ngày đích: 70% train, 15% validation, 15% test; các cửa sổ vẫn giữ đủ lịch sử 20 phiên.
- Validation dùng để early stopping, chọn checkpoint và tính prediction-based block ablation; test chỉ dùng để báo cáo dự báo ngoài mẫu.
- NGC dùng group-norm regularization với Adam. Đây không phải proximal group lasso tạo trọng số bằng 0 tuyệt đối; cạnh được chọn bằng mức tăng validation loss khi ablate toàn bộ một nguồn, sau đó áp dụng ngưỡng quantile 0.70.
- VAR đa biến dùng 20 lag, kiểm định chung theo cặp nguồn–đích và Benjamini–Hochberg FDR.
- Độ ổn định được đo trên 20 seed (`42…61`), ngưỡng cạnh ổn định là tần suất `>= 0.80`.
- Robustness gồm lịch ngày giao dịch chung và biến thể dịch thêm một phiên cho đầu vào thị trường toàn cầu.

## Dữ liệu và quy tắc thời gian

Pipeline chính lấy lịch VN-Index làm trục. Các thị trường đóng cửa được forward-fill có giới hạn; SSE có thêm nội suy giới hạn đã được ghi trong audit. BTC không được fill. Log return chỉ được tính khi cả `P_t` và `P_(t-1)` đều dương.

Giá WTI `-37.63` ngày `2020-04-20` được giữ trong bảng giá để truy xuất nguồn gốc, nhưng return tại `2020-04-20` và `2020-04-21` bị đánh dấu không hợp lệ. Pipeline không nối tắt qua mức giá âm.

Chi tiết dữ liệu và biên chia mẫu nằm tại:

- `results/camera_ready/data_audit.md`
- `results/camera_ready/tables/main_split_summary.csv`
- `results/camera_ready/tables/invalid_return_endpoints.csv`

## Cấu trúc hiện hành

```text
config.py                            cấu hình và hợp đồng thí nghiệm
src/camera_ready_data.py             dựng panel, return và split
src/camera_ready_models.py           LSTM, NGC và block ablation
src/camera_ready_analysis.py         VAR, kiểm định, graph và GBRI
scripts/14_camera_ready_revision.py  pipeline chính
scripts/15_verify_camera_ready_outputs.py  kiểm tra tính toàn vẹn
tests/test_camera_ready.py           unit tests
results/camera_ready/                dữ liệu, bảng, hình và checkpoint chính thức
```

Các đầu ra cần dùng khi sửa paper:

- `results/camera_ready/camera_ready_results_summary.md`
- `results/camera_ready/tables/forecast_metrics.csv`
- `results/camera_ready/tables/forecast_inference.json`
- `results/camera_ready/tables/edge_stability_20seeds.csv`
- `results/camera_ready/tables/node_role_stability_20seeds.csv`
- `results/camera_ready/tables/calendar_timing_robustness.csv`
- `results/camera_ready/tables/var_ngc_graph_comparison.json`
- `results/camera_ready/tables/gbri_prediction_weights.csv`

## Ý nghĩa của `max_abs_repeat_diff = 0.0`

Giá trị `0.0` khi chạy lại **cùng dữ liệu, cùng seed, cùng thiết bị và cùng phiên bản thư viện** là kết quả mong muốn: cấu hình deterministic đã tái tạo đúng từng dự báo. Nó không có nghĩa mô hình luôn cho một kết quả với mọi seed, cũng không có nghĩa dữ liệu hoặc mô hình bị hằng.

Kiểm tra độ nhạy giữa các seed được báo riêng trong `edge_stability_20seeds.csv` và `node_role_stability_20seeds.csv`. Không nên kỳ vọng khớp bit tuyệt đối khi đổi GPU, driver, PyTorch hoặc kiến trúc phần cứng.

## Pipeline cũ

Để tái tạo kết quả lịch sử, có thể chạy:

```bash
python run_pipeline.py --profile legacy
```

Luồng này gồm `scripts/00_*` đến `scripts/12_*`, `src/neural_granger.py`, `data/processed/`, `results/tables/` và `results/figures/`. Nó không được gọi bởi pipeline mặc định. `run_full_gpu.py` hiện chỉ là wrapper tương thích và đã chuyển sang pipeline camera-ready.
