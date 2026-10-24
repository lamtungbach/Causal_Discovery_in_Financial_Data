from pathlib import Path
import sys
import subprocess
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import TABLE_DIR
from src.common import save_table


def main():
    """Ablation nhẹ. Chạy nhiều cấu hình Neural Granger có thể tốn thời gian.
    Mặc định chỉ ghi thiết kế thí nghiệm; bạn có thể bật run=True để chạy thật.
    """
    designs = []
    for lags in [3, 5, 10]:
        for lam in [1e-4, 1e-3, 1e-2]:
            designs.append({"experiment": f"ng_lag{lags}_lam{lam}", "lags": lags, "lambda": lam, "purpose": "Sensitivity of lag length and sparsity regularization"})
    designs.extend([
        {"experiment": "without_wavelet", "lags": 5, "lambda": 1e-3, "purpose": "Neural Granger on raw returns"},
        {"experiment": "with_wavelet_bands", "lags": 5, "lambda": 1e-3, "purpose": "Neural Granger on multi-scale representation"},
        {"experiment": "drop_vix", "lags": 5, "lambda": 1e-3, "purpose": "Check importance of fear index"},
        {"experiment": "drop_btc", "lags": 5, "lambda": 1e-3, "purpose": "Check crypto contribution"},
    ])
    save_table(pd.DataFrame(designs), TABLE_DIR / "12_ablation_design.csv")
    print("✓ Ablation design exported. Chạy thủ công từng cấu hình trong scripts/08_neural_granger.py nếu cần.")

if __name__ == "__main__":
    main()
