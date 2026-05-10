"""
src/evaluation/metrics.py
SHD, Precision, Recall, F1, FPR/TPR.

Fix: compute_metrics() trả về MetricResult (dataclass) thay vì dict
     → test_metrics.py dùng isinstance(m, MetricResult) đúng
     MetricResult có đầy đủ tp, fp, fn, tn, fpr, shd
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MetricResult:
    """Kết quả đánh giá một phương pháp — đầy đủ để dùng trong tests."""
    shd      : float
    precision: float
    recall   : float
    f1       : float
    fpr      : float
    tp       : int = 0
    fp       : int = 0
    fn       : int = 0
    tn       : int = 0

    def to_dict(self) -> dict:
        return {
            "SHD"          : round(self.shd,       4),
            "Precision"    : round(self.precision,  4),
            "Recall (TPR)" : round(self.recall,     4),
            "F1-score"     : round(self.f1,         4),
            "FPR"          : round(self.fpr,        4),
            "TP"           : self.tp,
            "FP"           : self.fp,
            "FN"           : self.fn,
            "TN"           : self.tn,
        }


def structural_hamming_distance(
    true_dag: np.ndarray,
    pred_dag: np.ndarray,
) -> int:
    """
    Structural Hamming Distance (SHD).

    Đếm số thao tác tối thiểu để chuyển pred → true:
      - Thêm cạnh bị thiếu     (FN)
      - Xóa cạnh thừa          (FP)
      - Đảo ngược cạnh sai hướng (reversal — tính 1 lần, không phải 2)

    Parameters
    ----------
    true_dag, pred_dag : np.ndarray (d, d), binary {0, 1}
    """
    true = (true_dag != 0).astype(int)
    pred = (pred_dag != 0).astype(int)
    d    = true.shape[0]

    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())

    # Reversal: pred[i,j]=1 & true[j,i]=1 → sai hướng, tính 1 lần
    reversal = 0
    for i in range(d):
        for j in range(i + 1, d):
            if pred[i, j] == 1 and true[j, i] == 1:
                reversal += 1
                fp       -= 1
                fn       -= 1
            elif pred[j, i] == 1 and true[i, j] == 1:
                reversal += 1
                fp       -= 1
                fn       -= 1

    return max(int(fp + fn + reversal), 0)


def compute_metrics(
    true_dag: np.ndarray,
    pred_dag: np.ndarray,
) -> MetricResult:
    """
    Tính TP/FP/FN/TN và tất cả metrics dẫn xuất.

    Bỏ qua diagonal (không có self-loop).

    Parameters
    ----------
    true_dag, pred_dag : np.ndarray (d, d), binary {0, 1}

    Returns
    -------
    MetricResult  — dataclass với precision, recall, f1, fpr, shd, tp, fp, fn, tn
    """
    d    = true_dag.shape[0]
    mask = ~np.eye(d, dtype=bool)   # bỏ diagonal

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
    shd = structural_hamming_distance(true_dag, pred_dag)

    return MetricResult(
        shd       = shd,
        precision = precision,
        recall    = recall,
        f1        = f1,
        fpr       = fpr,
        tp        = tp,
        fp        = fp,
        fn        = fn,
        tn        = tn,
    )