"""
experiments/threshold_sensitivity.py
Tune ngưỡng threshold của NOTEARS — chọn giá trị tối ưu.

Đặt file tại: experiments/threshold_sensitivity.py

Mục tiêu:
  Tìm threshold w* sao cho DAG:
    1. Đủ thưa    (loại bỏ cạnh nhiễu)
    2. Đủ đầy đủ  (giữ lại cạnh có ý nghĩa)
    3. F1 cao nhất so với Granger pseudo-ground-truth

Chạy:
    python experiments/threshold_sensitivity.py
    python experiments/threshold_sensitivity.py --lambda1 0.05 --bootstrap 500
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    NODE_NAMES, NOTEARS_THRESHOLDS, NOTEARS_LAMBDA1,
    FIGURES_DIR, TABLES_DIR, BOOTSTRAP_N, GRANGER_MAX_LAG,
)
from src.utils.logger  import setup_logger, get_logger, LogTimer
from src.utils.io_utils import ResultIO

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core experiment class
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdSensitivityExperiment:
    """
    Thực nghiệm tune threshold cho NOTEARS.

    Quy trình:
      1. Tải dữ liệu volatility (từ snapshot hoặc chạy lại preprocessor)
      2. Chạy Granger → lấy pseudo-ground-truth
      3. Chạy NOTEARS với lambda1 cố định → lấy W_raw
      4. Áp dụng từng threshold trong dải thresholds
      5. Tính SHD, Precision, Recall, F1 + Bootstrap CI cho mỗi threshold
      6. Vẽ plots và lưu kết quả

    Parameters
    ----------
    thresholds   : list[float]  dải threshold cần thử
    lambda1      : float        L1 regularization của NOTEARS
    n_bootstrap  : int          số lần bootstrap CI
    output_dir   : str          thư mục lưu kết quả
    """

    def __init__(
        self,
        thresholds  : list  = None,
        lambda1     : float = NOTEARS_LAMBDA1,
        n_bootstrap : int   = BOOTSTRAP_N,
        output_dir  : str   = "reports",
    ):
        self.thresholds  = thresholds or NOTEARS_THRESHOLDS
        self.lambda1     = lambda1
        self.n_bootstrap = n_bootstrap
        self.io          = ResultIO(output_dir)

        # State
        self.X_           = None
        self.node_names_  = None
        self.vol_df_      = None
        self.W_raw_       = None    # Ma trận W trước threshold
        self.granger_dag_ = None    # Pseudo-ground-truth
        self.results_df_  = None    # Bảng kết quả theo threshold

    # =========================================================================
    # Setup
    # =========================================================================

    def load_data(
        self,
        vnindex_path: str = None,
        from_snapshot: bool = True,
    ) -> None:
        """
        Tải dữ liệu X, vol_df, node_names.

        Ưu tiên load từ snapshot (nhanh hơn) nếu có.
        Fallback: chạy lại DataCollector + DataPreprocessor.
        """
        if from_snapshot:
            try:
                data = self.io.load_snapshot()
                vol_df = data.get("vol_df")
                if vol_df is not None:
                    from sklearn.preprocessing import StandardScaler
                    self.X_          = StandardScaler().fit_transform(vol_df.values)
                    self.vol_df_     = vol_df
                    self.node_names_ = list(vol_df.columns)
                    logger.info(f"Load từ snapshot OK | X={self.X_.shape}")
                    return
            except FileNotFoundError:
                logger.info("Không có snapshot → chạy lại preprocessor")

        # Chạy lại pipeline data
        from src.data.collector    import DataCollector
        from src.data.preprocessor import DataPreprocessor

        collector = DataCollector(vnindex_path=vnindex_path)
        prices    = collector.fetch()
        prep      = DataPreprocessor()
        X, vol    = prep.fit_transform(prices, standardize=True)

        self.X_          = X
        self.vol_df_     = prep.volatility_
        self.node_names_ = list(vol.columns) if hasattr(vol, "columns") \
                           else NODE_NAMES
        logger.info(f"Dữ liệu: X={self.X_.shape}")

    def run_granger_baseline(self) -> np.ndarray:
        """Chạy Granger để lấy pseudo-ground-truth DAG."""
        from src.models.granger import GrangerCausality

        logger.info("Chạy Granger Causality (pseudo-ground-truth)...")
        with LogTimer("Granger baseline", logger):
            model  = GrangerCausality(
                max_lag    = GRANGER_MAX_LAG,
                node_names = self.node_names_,
            )
            result = model.fit_timed(self.X_)

        self.granger_dag_ = result.binary_matrix
        logger.info(
            f"Granger edges: {result.n_edges} | "
            f"Hub: {result.hub_nodes(1)}"
        )
        return self.granger_dag_

    def run_notears_raw(self) -> np.ndarray:
        """Chạy NOTEARS 1 lần duy nhất → lưu W_raw (trước threshold)."""
        from src.models.notears import NOTEARS

        logger.info(f"Chạy NOTEARS (lambda1={self.lambda1}) → lấy W_raw...")
        with LogTimer("NOTEARS optimization", logger):
            model = NOTEARS(
                lambda1    = self.lambda1,
                threshold  = 0.0,       # threshold=0 để giữ toàn bộ W
                node_names = self.node_names_,
            )
            model.fit_timed(self.X_)

        self.W_raw_ = model.W_raw_
        logger.info(
            f"W_raw: min={self.W_raw_.min():.4f} | "
            f"max={self.W_raw_.max():.4f} | "
            f"mean_abs={np.abs(self.W_raw_).mean():.4f}"
        )
        return self.W_raw_

    # =========================================================================
    # Main experiment
    # =========================================================================

    def run(self) -> pd.DataFrame:
        """
        Chạy toàn bộ experiment threshold sensitivity.

        Returns
        -------
        pd.DataFrame — bảng kết quả theo threshold, dùng cho Chương 3.5.3
        """
        if self.X_ is None:
            raise RuntimeError("Gọi load_data() trước")
        if self.granger_dag_ is None:
            self.run_granger_baseline()
        if self.W_raw_ is None:
            self.run_notears_raw()

        logger.info(f"\nThử {len(self.thresholds)} threshold: {self.thresholds}")
        logger.info("=" * 60)

        from src.evaluation.bootstrap import bootstrap_ci

        rows = []
        for thr in self.thresholds:
            # Áp dụng threshold
            W_bin = (np.abs(self.W_raw_) > thr).astype(int)
            np.fill_diagonal(W_bin, 0)
            n_edges = int(W_bin.sum())
            d       = W_bin.shape[0]
            density = n_edges / (d * (d - 1)) if d > 1 else 0

            # Bootstrap CI
            boot = bootstrap_ci(
                true_dag     = self.granger_dag_,
                pred_dag     = W_bin,
                n_bootstrap  = self.n_bootstrap,
                random_state = 42,
            )

            row = {
                "Threshold"   : thr,
                "Edges"       : n_edges,
                "Density"     : round(density, 4),
                "SHD"         : boot.shd,
                "Precision"   : round(boot.precision, 4),
                "Recall"      : round(boot.recall, 4),
                "F1"          : round(boot.f1, 4),
                "FPR"         : round(boot.fpr, 4),
                "Prec CI"     : f"[{boot.prec_ci[0]:.3f}, {boot.prec_ci[1]:.3f}]",
                "Recall CI"   : f"[{boot.recall_ci[0]:.3f}, {boot.recall_ci[1]:.3f}]",
                "F1 CI"       : f"[{boot.f1_ci[0]:.3f}, {boot.f1_ci[1]:.3f}]",
            }
            rows.append(row)
            logger.info(
                f"  thr={thr:.2f} | edges={n_edges:2d} | "
                f"SHD={boot.shd:3d} | P={boot.precision:.3f} | "
                f"R={boot.recall:.3f} | F1={boot.f1:.3f}"
            )

        self.results_df_ = pd.DataFrame(rows)

        # Lưu kết quả
        self._save_results()
        # Vẽ plots
        self._plot_results()
        # In summary
        self._print_summary()

        return self.results_df_

    # =========================================================================
    # Analysis helpers
    # =========================================================================

    def best_threshold(self, metric: str = "F1") -> float:
        """
        Tìm threshold tối ưu theo metric chỉ định.

        Parameters
        ----------
        metric : str  "F1" | "Precision" | "Recall" | "SHD"

        Returns
        -------
        float  threshold tối ưu
        """
        if self.results_df_ is None:
            raise RuntimeError("Chạy run() trước")

        if metric == "SHD":
            idx = self.results_df_["SHD"].idxmin()
        else:
            idx = self.results_df_[metric].idxmax()

        best_row = self.results_df_.iloc[idx]
        logger.info(
            f"Best threshold ({metric}): "
            f"thr={best_row['Threshold']} | "
            f"F1={best_row['F1']:.4f} | "
            f"edges={int(best_row['Edges'])}"
        )
        return float(best_row["Threshold"])

    def elbow_threshold(self) -> float:
        """
        Tìm threshold theo phương pháp elbow (điểm gãy của đường edges vs threshold).
        Hữu ích khi không có ground truth tốt.

        Returns
        -------
        float  threshold tại điểm elbow
        """
        if self.results_df_ is None:
            raise RuntimeError("Chạy run() trước")

        thresholds = self.results_df_["Threshold"].values
        n_edges    = self.results_df_["Edges"].values.astype(float)

        # Tính second derivative để tìm điểm gãy
        d1    = np.diff(n_edges)
        d2    = np.diff(d1)
        elbow = thresholds[1:-1][np.argmax(np.abs(d2))]
        logger.info(f"Elbow threshold: {elbow}")
        return float(elbow)

    # =========================================================================
    # Save & Plot
    # =========================================================================

    def _save_results(self) -> None:
        """Lưu kết quả ra CSV và LaTeX."""
        csv_path = Path(TABLES_DIR) / "threshold_sensitivity_full.csv"
        self.results_df_.to_csv(csv_path, index=False)
        logger.info(f"CSV → {csv_path}")

        tex_path = Path(TABLES_DIR) / "threshold_sensitivity.tex"
        cols_tex = ["Threshold", "Edges", "Density", "SHD", "Precision", "Recall", "F1"]
        tex_str  = (
            self.results_df_[cols_tex]
            .to_latex(
                index   = False,
                float_format = "%.4f",
                caption = "NOTEARS — Phân tích độ nhạy Threshold (Chương 3.5.3)",
                label   = "tab:threshold_sensitivity",
            )
        )
        tex_path.write_text(tex_str, encoding="utf-8")
        logger.info(f"LaTeX → {tex_path}")

    def _plot_results(self) -> None:
        """Vẽ 3 plots: edges/density, F1 curve, Precision-Recall tradeoff."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            logger.warning("pip install matplotlib")
            return

        thresholds = self.results_df_["Threshold"].values
        n_edges    = self.results_df_["Edges"].values
        densities  = self.results_df_["Density"].values
        f1s        = self.results_df_["F1"].values
        precs      = self.results_df_["Precision"].values
        recs       = self.results_df_["Recall"].values
        shds       = self.results_df_["SHD"].values

        best_thr = self.best_threshold("F1")

        fig = plt.figure(figsize=(16, 5))
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        # ── Plot 1: Edges & Density ───────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        ax1b = ax1.twinx()
        ax1.plot(thresholds, n_edges, "b-o", lw=2, ms=7, label="Edges")
        ax1b.plot(thresholds, densities, "r--s", lw=1.5, ms=6, label="Density")
        ax1.axvline(best_thr, color="green", ls=":", lw=1.8, alpha=0.8)
        ax1.set_xlabel("Threshold", fontsize=11)
        ax1.set_ylabel("Số cạnh", color="blue", fontsize=10)
        ax1b.set_ylabel("Graph Density", color="red", fontsize=10)
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1b.tick_params(axis="y", labelcolor="red")
        ax1.set_title("Edges & Density", fontsize=11, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(thresholds, n_edges, alpha=0.08, color="blue")

        # ── Plot 2: F1 & SHD curve ────────────────────────────────────────────
        ax2  = fig.add_subplot(gs[1])
        ax2b = ax2.twinx()
        ax2.plot(thresholds, f1s, "g-^", lw=2, ms=7, label="F1-score")
        ax2b.plot(thresholds, shds, "m--v", lw=1.5, ms=6, label="SHD")
        ax2.axvline(best_thr, color="green", ls=":", lw=1.8, alpha=0.8)
        ax2.annotate(
            f"Best\nthr={best_thr}",
            xy=(best_thr, max(f1s)),
            xytext=(best_thr + 0.03, max(f1s) * 0.95),
            fontsize=8, color="green",
            arrowprops=dict(arrowstyle="->", color="green", lw=1.2),
        )
        ax2.set_xlabel("Threshold", fontsize=11)
        ax2.set_ylabel("F1-score", color="green", fontsize=10)
        ax2b.set_ylabel("SHD ↓", color="purple", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="green")
        ax2b.tick_params(axis="y", labelcolor="purple")
        ax2.set_title("F1-score & SHD", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # ── Plot 3: Precision-Recall tradeoff ─────────────────────────────────
        ax3 = fig.add_subplot(gs[2])
        sc  = ax3.scatter(recs, precs, c=thresholds, cmap="viridis",
                          s=80, zorder=5)
        ax3.plot(recs, precs, "gray", lw=1, alpha=0.5, zorder=4)
        # Ghi nhãn threshold lên từng điểm
        for thr, r, p in zip(thresholds, recs, precs):
            ax3.annotate(
                f"{thr}", (r, p),
                textcoords="offset points", xytext=(4, 4),
                fontsize=7, color="dimgray",
            )
        plt.colorbar(sc, ax=ax3, label="Threshold")
        ax3.set_xlabel("Recall", fontsize=11)
        ax3.set_ylabel("Precision", fontsize=11)
        ax3.set_title("Precision–Recall Tradeoff", fontsize=11, fontweight="bold")
        ax3.set_xlim(-0.05, 1.05)
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, alpha=0.3)

        fig.suptitle(
            f"NOTEARS — Phân tích độ nhạy Threshold (λ₁={self.lambda1})\n"
            f"Pseudo-ground-truth: Granger Causality",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()

        out = Path(FIGURES_DIR) / "threshold_sensitivity_full.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot → {out}")

    def _print_summary(self) -> None:
        """In bảng kết quả ra console."""
        df = self.results_df_

        print("\n" + "═" * 70)
        print("  THRESHOLD SENSITIVITY — NOTEARS")
        print(f"  λ₁={self.lambda1} | Bootstrap n={self.n_bootstrap}")
        print("═" * 70)
        cols_print = ["Threshold", "Edges", "Density", "SHD", "Precision", "Recall", "F1"]
        print(df[cols_print].to_string(index=False))
        print("─" * 70)

        best_f1  = self.best_threshold("F1")
        best_shd = self.best_threshold("SHD")
        print(f"  ★ Best threshold (F1) : {best_f1}")
        print(f"  ★ Best threshold (SHD): {best_shd}")
        print("═" * 70 + "\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="NOTEARS Threshold Sensitivity Experiment"
    )
    p.add_argument(
        "--lambda1", type=float, default=NOTEARS_LAMBDA1,
        help=f"L1 regularization (mặc định: {NOTEARS_LAMBDA1})"
    )
    p.add_argument(
        "--thresholds", nargs="+", type=float,
        default=NOTEARS_THRESHOLDS,
        help="Danh sách threshold cần thử (cách nhau bởi dấu cách)"
    )
    p.add_argument(
        "--bootstrap", type=int, default=BOOTSTRAP_N,
        help=f"Số lần bootstrap CI (mặc định: {BOOTSTRAP_N})"
    )
    p.add_argument(
        "--vnindex", type=str, default=None,
        help="Đường dẫn file VN-Index CSV"
    )
    p.add_argument(
        "--no-snapshot", action="store_true",
        help="Không dùng snapshot, chạy lại preprocessor"
    )
    return p.parse_args()


if __name__ == "__main__":
    setup_logger("KLTN")
    args = parse_args()

    logger.info("=" * 60)
    logger.info("EXPERIMENT: Threshold Sensitivity")
    logger.info("=" * 60)
    logger.info(f"  lambda1    = {args.lambda1}")
    logger.info(f"  thresholds = {args.thresholds}")
    logger.info(f"  bootstrap  = {args.bootstrap}")

    exp = ThresholdSensitivityExperiment(
        thresholds  = args.thresholds,
        lambda1     = args.lambda1,
        n_bootstrap = args.bootstrap,
    )

    exp.load_data(
        vnindex_path  = args.vnindex,
        from_snapshot = not args.no_snapshot,
    )

    results_df = exp.run()

    print(f"\nBest threshold (F1) : {exp.best_threshold('F1')}")
    print(f"Best threshold (SHD): {exp.best_threshold('SHD')}")
    print(f"Elbow threshold     : {exp.elbow_threshold()}")