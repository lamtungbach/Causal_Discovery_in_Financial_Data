"""
src/evaluation/bootstrap.py
Bootstrap Confidence Interval cho SHD, Precision, Recall, F1.

Đặt file tại: src/evaluation/bootstrap.py
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data class kết quả
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BootstrapResult:
    """
    Kết quả đánh giá kèm 95% Confidence Interval.

    Attributes
    ----------
    shd, precision, recall, f1, fpr : float   — điểm số gốc (không bootstrap)
    prec_ci, recall_ci, f1_ci, shd_ci         — khoảng tin cậy (lower, upper)
    n_bootstrap : int                          — số lần lặp bootstrap
    """
    shd        : float
    precision  : float
    recall     : float
    f1         : float
    fpr        : float

    prec_ci    : Tuple[float, float]
    recall_ci  : Tuple[float, float]
    f1_ci      : Tuple[float, float]
    shd_ci     : Tuple[float, float]

    n_bootstrap: int

    # ── Helpers ───────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Chuyển sang dict để lưu CSV / in báo cáo."""
        return {
            "SHD"             : round(self.shd, 4),
            "SHD CI 95%"      : f"[{self.shd_ci[0]:.1f}, {self.shd_ci[1]:.1f}]",
            "Precision"       : round(self.precision, 4),
            "Precision CI 95%": f"[{self.prec_ci[0]:.3f}, {self.prec_ci[1]:.3f}]",
            "Recall (TPR)"    : round(self.recall, 4),
            "Recall CI 95%"   : f"[{self.recall_ci[0]:.3f}, {self.recall_ci[1]:.3f}]",
            "F1-score"        : round(self.f1, 4),
            "F1 CI 95%"       : f"[{self.f1_ci[0]:.3f}, {self.f1_ci[1]:.3f}]",
            "FPR"             : round(self.fpr, 4),
        }

    def format_table_row(self, method_name: str, runtime: float) -> dict:
        """
        Format 1 dòng cho bảng so sánh Chương 4.5.

        Ví dụ output:
            Method      | SHD ↓       | Precision ↑            | ...
            NOTEARS     | 3 [2,4]     | 0.800 [0.700, 0.867]   | ...
        """
        return {
            "Method"      : method_name,
            "SHD ↓"       : f"{int(self.shd)} [{self.shd_ci[0]:.0f}, {self.shd_ci[1]:.0f}]",
            "Precision ↑" : f"{self.precision:.3f} [{self.prec_ci[0]:.3f}, {self.prec_ci[1]:.3f}]",
            "Recall ↑"    : f"{self.recall:.3f} [{self.recall_ci[0]:.3f}, {self.recall_ci[1]:.3f}]",
            "F1 ↑"        : f"{self.f1:.3f} [{self.f1_ci[0]:.3f}, {self.f1_ci[1]:.3f}]",
            "FPR ↓"       : f"{self.fpr:.3f}",
            "Runtime (s)" : f"{runtime:.2f}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Hàm metrics nội bộ (tránh circular import)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics_internal(
    true_dag: np.ndarray,
    pred_dag: np.ndarray,
) -> dict:
    """
    Tính TP/FP/FN/TN và các metrics dẫn xuất.
    Bỏ qua diagonal (không có self-loop).
    """
    d    = true_dag.shape[0]
    mask = ~np.eye(d, dtype=bool)

    true = (true_dag[mask] != 0).astype(int)
    pred = (pred_dag[mask] != 0).astype(int)

    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # SHD (đơn giản: FP + FN, không tính reversal riêng)
    shd = fp + fn
    for i in range(d):
        for j in range(i + 1, d):
            if pred_dag[i, j] == 1 and true_dag[j, i] == 1:
                shd -= 1   # reversal tính 1 lần
            elif pred_dag[j, i] == 1 and true_dag[i, j] == 1:
                shd -= 1

    return {
        "precision": precision,
        "recall"   : recall,
        "f1"       : f1,
        "fpr"      : fpr,
        "shd"      : max(shd, 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hàm bootstrap chính
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    true_dag    : np.ndarray,
    pred_dag    : np.ndarray,
    n_bootstrap : int   = 1000,
    ci          : float = 0.95,
    random_state: int   = 42,
) -> BootstrapResult:
    """
    Tính Bootstrap Confidence Interval cho các metrics causal discovery.

    Strategy
    --------
    Resample d nodes (with replacement) → tạo submatrix d×d → tính metrics.
    Lặp n_bootstrap lần → lấy quantile [α/2, 1-α/2].

    Parameters
    ----------
    true_dag     : np.ndarray (d, d)  ground truth DAG (binary)
    pred_dag     : np.ndarray (d, d)  predicted DAG (binary)
    n_bootstrap  : int                số lần bootstrap (khuyến nghị ≥ 1000)
    ci           : float              mức tin cậy, mặc định 0.95
    random_state : int                seed để tái tạo kết quả

    Returns
    -------
    BootstrapResult  — metrics gốc + CI 95%

    Example
    -------
    >>> result = bootstrap_ci(true_dag, notears_binary, n_bootstrap=1000)
    >>> print(result.to_dict())
    """
    rng   = np.random.RandomState(random_state)
    d     = true_dag.shape[0]
    alpha = (1 - ci) / 2

    precs, recs, f1s, shds = [], [], [], []

    logger.info(f"Bootstrap CI: n={n_bootstrap}, ci={ci}, d={d}")

    for _ in range(n_bootstrap):
        idx  = rng.choice(d, size=d, replace=True)
        t_b  = true_dag[np.ix_(idx, idx)]
        p_b  = pred_dag[np.ix_(idx, idx)]
        m    = _compute_metrics_internal(t_b, p_b)
        precs.append(m["precision"])
        recs.append(m["recall"])
        f1s.append(m["f1"])
        shds.append(m["shd"])

    # Điểm số gốc (không bootstrap)
    base = _compute_metrics_internal(true_dag, pred_dag)

    def _ci(arr: list) -> Tuple[float, float]:
        return (
            round(float(np.quantile(arr, alpha)), 4),
            round(float(np.quantile(arr, 1 - alpha)), 4),
        )

    result = BootstrapResult(
        shd        = base["shd"],
        precision  = base["precision"],
        recall     = base["recall"],
        f1         = base["f1"],
        fpr        = base["fpr"],
        prec_ci    = _ci(precs),
        recall_ci  = _ci(recs),
        f1_ci      = _ci(f1s),
        shd_ci     = _ci(shds),
        n_bootstrap= n_bootstrap,
    )

    logger.info(
        f"  SHD={result.shd} | "
        f"P={result.precision:.3f}{list(result.prec_ci)} | "
        f"R={result.recall:.3f}{list(result.recall_ci)} | "
        f"F1={result.f1:.3f}{list(result.f1_ci)}"
    )
    return result