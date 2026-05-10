"""
src/evaluation/comparator.py
So sánh tổng hợp 3 phương pháp: Granger, PC Algorithm, NOTEARS.

Đặt file tại: src/evaluation/comparator.py
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from .bootstrap import bootstrap_ci, BootstrapResult

logger = logging.getLogger(__name__)


class MethodComparator:
    """
    So sánh nhiều phương pháp causal discovery trên cùng ground truth DAG.

    Trong khóa luận:
      - ground truth = Granger Causality (pseudo-ground-truth)
      - so sánh: PC Algorithm và NOTEARS

    Sử dụng
    -------
    >>> comparator = MethodComparator(true_dag=granger_result.binary_matrix)
    >>> comparator.add("PC Algorithm", pc_result)
    >>> comparator.add("NOTEARS",      notears_result)
    >>> df = comparator.comparison_table()
    >>> comparator.print_summary()
    """

    def __init__(
        self,
        true_dag    : np.ndarray,
        n_bootstrap : int   = 1000,
        ci          : float = 0.95,
        random_state: int   = 42,
    ):
        """
        Parameters
        ----------
        true_dag     : np.ndarray (d, d)  binary
            Ground truth DAG. Khóa luận dùng Granger làm pseudo-ground-truth.
        n_bootstrap  : int
            Số lần bootstrap để tính CI 95%. Khuyến nghị >= 1000.
        ci           : float
            Mức tin cậy. Mặc định 0.95.
        """
        self.true_dag     = true_dag
        self.n_bootstrap  = n_bootstrap
        self.ci           = ci
        self.random_state = random_state

        # Lưu CausalResult và BootstrapResult theo thứ tự add()
        self._results : Dict[str, object]          = {}
        self._boots   : Dict[str, BootstrapResult] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def add(self, name: str, result: object) -> None:
        """
        Thêm một phương pháp vào bộ so sánh và tính Bootstrap CI ngay.

        Parameters
        ----------
        name   : str           tên hiển thị (vd "NOTEARS", "PC Algorithm")
        result : CausalResult  kết quả từ model.fit_timed()
        """
        self._results[name] = result
        logger.info(f"[Comparator] Đánh giá '{name}' ...")

        boot = bootstrap_ci(
            true_dag    = self.true_dag,
            pred_dag    = result.binary_matrix,
            n_bootstrap = self.n_bootstrap,
            ci          = self.ci,
            random_state= self.random_state,
        )
        self._boots[name] = boot

    def comparison_table(self) -> pd.DataFrame:
        """
        Sinh bảng so sánh chuẩn cho Chương 4.5.

        Mỗi hàng = 1 phương pháp.
        Mỗi cột  = 1 metric kèm CI 95% trong ngoặc vuông.

        Returns
        -------
        pd.DataFrame  với index = Method name
        """
        rows = []
        for name, boot in self._boots.items():
            result = self._results[name]
            rows.append(
                boot.format_table_row(
                    method_name = name,
                    runtime     = result.runtime_seconds,
                )
            )
        df = pd.DataFrame(rows).set_index("Method")
        return df

    def raw_scores(self) -> pd.DataFrame:
        """
        Bảng điểm số thuần (không có CI) — tiện để vẽ bar chart.

        Returns
        -------
        pd.DataFrame  với các cột: SHD, Precision, Recall, F1, FPR, Runtime
        """
        rows = []
        for name, boot in self._boots.items():
            result = self._results[name]
            rows.append({
                "Method"     : name,
                "SHD"        : boot.shd,
                "Precision"  : round(boot.precision, 4),
                "Recall"     : round(boot.recall, 4),
                "F1"         : round(boot.f1, 4),
                "FPR"        : round(boot.fpr, 4),
                "Runtime (s)": round(result.runtime_seconds, 3),
                "Edges"      : result.n_edges,
            })
        return pd.DataFrame(rows).set_index("Method")

    def best_method(self, metric: str = "F1") -> str:
        """
        Trả về tên phương pháp tốt nhất theo metric chỉ định.

        Parameters
        ----------
        metric : str  "F1" | "Precision" | "Recall" | "SHD"
                      (SHD: nhỏ hơn tốt hơn; còn lại lớn hơn tốt hơn)
        """
        scores_df = self.raw_scores()
        if metric not in scores_df.columns:
            raise ValueError(f"Metric '{metric}' không tồn tại. Chọn: {list(scores_df.columns)}")
        if metric == "SHD":
            return str(scores_df[metric].idxmin())
        return str(scores_df[metric].idxmax())

    def latex_table(self, caption: str = None, label: str = None) -> str:
        """
        Export bảng so sánh dưới dạng LaTeX để dán vào khóa luận.

        Parameters
        ----------
        caption : str  tiêu đề bảng LaTeX
        label   : str  label để \\ref{} trong LaTeX

        Returns
        -------
        str  — LaTeX code sẵn dán vào .tex
        """
        df      = self.comparison_table()
        caption = caption or "So sánh hiệu suất 3 phương pháp Causal Discovery (Chương 4.5)"
        label   = label   or "tab:method_comparison"
        return df.to_latex(
            escape    = False,
            bold_rows = True,
            caption   = caption,
            label     = label,
        )

    def save(self, path: str) -> None:
        """Lưu bảng so sánh ra file CSV."""
        df = self.comparison_table()
        df.to_csv(path)
        logger.info(f"Đã lưu bảng so sánh → {path}")

    def print_summary(self) -> None:
        """In bảng so sánh + highlight phương pháp tốt nhất ra console."""
        df   = self.comparison_table()
        best = self.best_method("F1")

        print("\n" + "═" * 72)
        print("  BẢNG SO SÁNH 3 PHƯƠNG PHÁP  (Chương 4.5 — Khóa luận)")
        print("  Ground truth: Granger Causality (pseudo-ground-truth)")
        print("═" * 72)
        print(df.to_string())
        print("─" * 72)
        print(f"  ★ Phương pháp tốt nhất (F1):  {best}")
        print("═" * 72 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Bar chart so sánh metrics
    # ─────────────────────────────────────────────────────────────────────────

    def plot_comparison(
        self,
        output_path: Optional[str] = None,
        figsize    : tuple = (12, 5),
    ):
        """
        Vẽ grouped bar chart so sánh Precision, Recall, F1 kèm CI error bars.
        Dùng cho Figure trong Chương 4.5.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install matplotlib")

        methods = list(self._boots.keys())
        metrics = ["Precision", "Recall", "F1"]
        colors  = ["#4C9BE8", "#F4845F", "#57C278"]

        scores = {
            "Precision": [self._boots[m].precision  for m in methods],
            "Recall"   : [self._boots[m].recall     for m in methods],
            "F1"       : [self._boots[m].f1         for m in methods],
        }
        errors = {
            "Precision": [
                (self._boots[m].precision - self._boots[m].prec_ci[0],
                 self._boots[m].prec_ci[1] - self._boots[m].precision)
                for m in methods
            ],
            "Recall": [
                (self._boots[m].recall - self._boots[m].recall_ci[0],
                 self._boots[m].recall_ci[1] - self._boots[m].recall)
                for m in methods
            ],
            "F1": [
                (self._boots[m].f1 - self._boots[m].f1_ci[0],
                 self._boots[m].f1_ci[1] - self._boots[m].f1)
                for m in methods
            ],
        }

        x     = np.arange(len(methods))
        width = 0.22
        fig, ax = plt.subplots(figsize=figsize)

        for k, (metric, color) in enumerate(zip(metrics, colors)):
            err = np.array(errors[metric]).T   # shape (2, n_methods)
            ax.bar(
                x + k * width,
                scores[metric],
                width,
                yerr    = err,
                capsize = 5,
                label   = metric,
                color   = color,
                alpha   = 0.85,
                error_kw= {"elinewidth": 1.5, "ecolor": "dimgray"},
            )

        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(
            "So sánh Precision / Recall / F1 (kèm Bootstrap CI 95%)\n3 Phương pháp Causal Discovery",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        if output_path:
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Đã lưu: {output_path}")

        return fig