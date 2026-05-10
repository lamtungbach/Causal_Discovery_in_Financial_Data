"""
run_all.py — Chạy TOÀN BỘ pipeline từ đầu đến cuối.

Thứ tự thực hiện:
  Bước 1: Thu thập & xử lý dữ liệu (DataPreprocessor)
  Bước 2: Chạy 3 mô hình (Neural Granger cMLP+cLSTM, PC Algorithm, NOTEARS)
  Bước 3: So sánh & đánh giá (SHD, Precision, Recall, F1)
  Bước 4: Risk Map & Phân tích Subperiod (Các giai đoạn khủng hoảng)

Cách chạy:
  python run_all.py             ← chạy tất cả các bước (KHUYẾN NGHỊ)
  python run_all.py --step 1    ← chỉ chạy bước 1
  python run_all.py --step 2    ← chỉ chạy bước 2
"""

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import setup_logger
logger = setup_logger("KLTN")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers in-display
# ─────────────────────────────────────────────────────────────────────────────

def header(text: str):
    print("\n" + "═" * 70)
    print(f"  {text}")
    print("═" * 70)

def step_header(n: int, text: str):
    print(f"\n{'─' * 70}")
    print(f"  BƯỚC {n}: {text}")
    print(f"{'─' * 70}")


# ─────────────────────────────────────────────────────────────────────────────
# Config tập trung — chỉnh ở đây thay vì sửa từng file
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Dữ liệu
    "data": {
        "raw_dir"          : "data/raw",
        "processed_dir"    : "data/processed",
        "vnindex_path"     : "data/raw/vnindex_manual.csv",
        "volatility_window": 21,
        "start"            : "2018-01-01",
        "end"              : "2024-12-31",
    },
    # Neural Granger Causality (PRIMARY)
    "neural_granger": {
        "lag"         : 1,
        "architecture": "both",          # cMLP + cLSTM
        "hidden_sizes": [32, 16],
        "lstm_hidden" : 32,
        "lam"         : 0.05,
        "lr"          : 1e-3,
        "max_epochs"  : 500,
        "patience"    : 50,
        "threshold"   : 0.3,
        "device"      : "cpu",
        "verbose"     : False,
    },
    # PC Algorithm (comparison)
    "pc": {
        "alpha": 0.05,
    },
    # NOTEARS (comparison)
    "notears": {
        "lambda1"  : 0.1,
        "threshold": 0.3,
        "max_iter" : 100,
    },
    # Bootstrap CI
    "bootstrap": {
        "n": 200,
    },
    # Early Warning System
    "ews": {
        "threshold_z": 2.0,
        "window"     : 21,
    },
    # Output
    "output_dir": "reports",
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Đọc tham số dòng lệnh ────────────────────────────────────────────────
    only_step = None
    if "--step" in sys.argv:
        idx = sys.argv.index("--step")
        try:
            only_step = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("❌ Dùng: python run_all.py --step [1|2|3|4]")
            sys.exit(1)

    header("NEURAL GRANGER CAUSALITY — CAUSAL DISCOVERY & RISK SPILLOVER")
    print(f"  {'Chạy tất cả các bước (Full Pipeline)' if only_step is None else f'Chỉ chạy Bước {only_step}'}\n")
    start_total = time.time()

    # ── Khởi tạo Pipeline ────────────────────────────────────────────────────
    try:
        from src.pipeline.full_pipeline import FullPipeline
        pipeline = FullPipeline(config=CONFIG)
        logger.info("✅ Pipeline khởi tạo thành công")
    except Exception as e:
        logger.error(f"❌ Lỗi khởi tạo Pipeline: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════════
    # BƯỚC 1 — THU THẬP & XỬ LÝ DỮ LIỆU
    # ══════════════════════════════════════════════════════════════════════════
    if only_step in (None, 1):
        step_header(1, "Thu thập & Xử lý Chuỗi thời gian (Volatility)")
        t = time.time()
        try:
            pipeline.step_b1_load_data()
            pipeline.step_b2_stationarity()
            print(f"\n  ✅ Bước 1 hoàn thành ({time.time() - t:.1f}s)")
        except Exception as e:
            logger.error(f"❌ Bước 1 lỗi: {e}")
            traceback.print_exc()
            if only_step == 1:
                sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════════
    # BƯỚC 2 — CHẠY 3 MÔ HÌNH NHÂN QUẢ
    # ══════════════════════════════════════════════════════════════════════════
    if only_step in (None, 2):
        step_header(2, "Học Cấu trúc Nhân quả (Causal Discovery)")

        # Nếu chạy riêng bước 2, cần load data trước
        if only_step == 2 and pipeline.data_ is None:
            logger.info("  Auto-load data cho bước 2...")
            try:
                pipeline.step_b1_load_data()
            except Exception as e:
                logger.error(f"❌ Không thể load data: {e}")
                sys.exit(1)

        t = time.time()
        try:
            print("\n  [2.1] Chạy Neural Granger Causality (cMLP + cLSTM) — PRIMARY...")
            pipeline.step_b3_neural_granger()

            print("\n  [2.2] Chạy PC Algorithm — Comparison...")
            pipeline.step_b4_pc_algorithm()

            print("\n  [2.3] Chạy NOTEARS — Comparison...")
            pipeline.step_b5_notears()

            print(f"\n  ✅ Bước 2 hoàn thành ({time.time() - t:.1f}s)")
        except Exception as e:
            logger.error(f"❌ Bước 2 lỗi: {e}")
            traceback.print_exc()
            if only_step == 2:
                sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════════
    # BƯỚC 3 — ĐÁNH GIÁ (SHD, F1, Bootstrap CI)
    # ══════════════════════════════════════════════════════════════════════════
    if only_step in (None, 3):
        step_header(3, "So sánh Hiệu suất Thuật toán (SHD, F1-Score, Bootstrap CI)")

        if only_step == 3 and not pipeline.results_:
            logger.error("❌ Bước 3 cần kết quả từ Bước 2. Chạy: python run_all.py --step 2 trước.")
            sys.exit(1)

        t = time.time()
        try:
            pipeline.step_b6_evaluate()
            print(f"\n  ✅ Bước 3 hoàn thành ({time.time() - t:.1f}s)")
        except Exception as e:
            logger.error(f"❌ Bước 3 lỗi: {e}")
            traceback.print_exc()
            if only_step == 3:
                sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════════
    # BƯỚC 4 — RISK MAP & SUBPERIOD ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    if only_step in (None, 4):
        step_header(4, "Risk Map & Phân tích Giai đoạn Khủng hoảng (Subperiod)")

        if only_step == 4 and pipeline.data_ is None:
            logger.error("❌ Bước 4 cần data. Chạy: python run_all.py --step 1 trước.")
            sys.exit(1)

        t = time.time()
        try:
            # 4.1 Risk Scores + Early Warning System
            print("\n  [4.1] Tính Risk Scores & Early Warning System...")
            pipeline.step_b7_risk_and_ews()

            # 4.2 Subperiod Analysis
            print("\n  [4.2] Phân tích theo giai đoạn (6 subperiods)...")
            _run_subperiod(pipeline)

            # 4.3 Save & Visualise
            print("\n  [4.3] Lưu kết quả & tạo biểu đồ...")
            pipeline.step_b8_save_and_visualise()

            print(f"\n  ✅ Bước 4 hoàn thành ({time.time() - t:.1f}s)")
        except Exception as e:
            logger.error(f"❌ Bước 4 lỗi: {e}")
            traceback.print_exc()

    # ── Tổng kết ─────────────────────────────────────────────────────────────
    elapsed = time.time() - start_total
    header(f"🎉 XONG! Tổng thời gian chạy: {elapsed:.1f} giây")
    _print_output_guide()


# ─────────────────────────────────────────────────────────────────────────────
# Subperiod helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_subperiod(pipeline):
    """Chạy SubperiodPipeline với data và config từ FullPipeline."""
    try:
        import pandas as pd
        import traceback
        from src.pipeline.subperiod_pipeline import SubperiodPipeline

        if pipeline.data_ is None or pipeline.variable_names_ is None:
            logger.warning("⚠️  Không có data cho subperiod — bỏ qua.")
            return

        import os
        if getattr(pipeline, '_vol_df', None) is not None:
            vol_df = pipeline._vol_df
        else:
            vol_path = os.path.join(
                pipeline.config["data"].get("processed_dir", "data/processed"),
                "volatility_21d.csv"
            )
            if os.path.exists(vol_path):
                vol_df = pd.read_csv(vol_path, index_col=0, parse_dates=True)
            else:
                logger.warning("⚠️  Không tìm thấy volatility_21d.csv để chạy subperiod.")
                return

        subpipe = SubperiodPipeline(
            vol_df     = vol_df,
            node_names = pipeline.variable_names_,
            method     = "neural",
            threshold  = pipeline.config.get("neural_granger", {}).get("threshold", 0.1),
        )
        results = subpipe.run()
        subpipe.print_summary()

    except Exception as e:
        logger.warning(f"⚠️  Subperiod analysis lỗi: {e}")
        import traceback
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Output guide
# ─────────────────────────────────────────────────────────────────────────────

def _print_output_guide():
    print("""
  Kết quả đã được xuất tự động ra thư mục:
    📂 reports/              ← Thư mục gốc
    📂 reports/subperiods/   ← Phân tích 6 giai đoạn

  Hướng dẫn sử dụng cho Khóa luận:
    Chương 4.1 (EDA)            ← data/results/summary_stats.csv
    Chương 4.2 (Stationarity)   ← data/results/adf_kpss_results.csv
    Chương 4.3 (Neural GC DAG)  ← reports/adj_NeuralGC_Ensemble.csv
    Chương 4.4 (So sánh)        ← reports/metrics.json
    Chương 4.5 (Risk Scores)    ← reports/risk_scores.json
    Chương 4.6 (Subperiod)      ← reports/subperiods/subperiod_summary.csv
    Chương 4.7 (EWS)            ← reports/ews_alerts.json
""")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()