"""
cleanup_old_results.py
======================
Xóa các file output cũ liên quan đến Granger Causality truyền thống.
Chạy 1 lần trước khi chạy pipeline mới.

Chạy: python cleanup_old_results.py
"""

import os
import shutil
from pathlib import Path

# File cần xóa trong data/results/
OLD_RESULT_FILES = [
    "granger_dag.csv",
    "granger_pvalues.csv",
    "notears_dag.csv",        # sẽ được tạo lại bởi pipeline mới
    "notears_weights.csv",
    "pc_dag.csv",
    "comparison_table.csv",   # sẽ được tạo lại
]

# File cần xóa trong data/processed/
OLD_PROCESSED_FILES = [
    "volatility_stationary.csv",   # ← ĐÂY LÀ VẤN ĐỀ CHÍNH (sai phân sai)
]

# File/folder trong src/models/ cần xóa
OLD_MODEL_FILES = [
    "src/models/granger.py",   # thay bằng neural_granger.py
]


def cleanup(base_dir: str = "."):
    base = Path(base_dir)
    removed = []
    not_found = []

    print("\n🧹 Dọn dẹp file cũ liên quan đến Granger Causality...")
    print("=" * 55)

    # data/results/
    results_dir = base / "data" / "results"
    for fname in OLD_RESULT_FILES:
        fpath = results_dir / fname
        if fpath.exists():
            fpath.unlink()
            removed.append(str(fpath))
            print(f"   ❌ Đã xóa: {fpath}")
        else:
            not_found.append(str(fpath))

    # data/processed/
    processed_dir = base / "data" / "processed"
    for fname in OLD_PROCESSED_FILES:
        fpath = processed_dir / fname
        if fpath.exists():
            fpath.unlink()
            removed.append(str(fpath))
            print(f"   ❌ Đã xóa: {fpath}  ← (volatile_stationary = sai phân sai)")
        else:
            not_found.append(str(fpath))

    # src/models/granger.py
    for frel in OLD_MODEL_FILES:
        fpath = base / frel
        if fpath.exists():
            fpath.unlink()
            removed.append(str(fpath))
            print(f"   ❌ Đã xóa: {fpath}  ← (thay bằng neural_granger.py)")
        else:
            not_found.append(str(fpath))

    print(f"\n   ✅ Đã xóa   : {len(removed)} file")
    print(f"   ℹ️  Không có : {len(not_found)} file (đã xóa trước hoặc không tồn tại)")

    print(f"\n   📋 File MỚI đã sẵn sàng:")
    print(f"      src/models/neural_granger.py   ← PRIMARY method")
    print(f"      src/models/pc_algorithm.py     ← comparison")
    print(f"      src/models/notears.py          ← comparison")
    print(f"      src/data/downloader.py         ← fixed VNIndex missing")
    print(f"      src/data/preprocessor.py       ← fixed volatility_21d.csv")
    print(f"      src/data/validator.py          ← no auto-differencing")

    print(f"\n   🚀 Bước tiếp theo:")
    print(f"      1. python src/data/downloader.py     # tải lại data")
    print(f"      2. python src/data/preprocessor.py   # tính volatility")
    print(f"      3. python src/data/validator.py      # kiểm định")
    print(f"      4. python run_all.py                 # chạy pipeline")


if __name__ == "__main__":
    cleanup(".")
