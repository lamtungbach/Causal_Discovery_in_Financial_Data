"""
src/pipeline/subperiod_pipeline.py
Phân tích cấu trúc nhân quả qua các giai đoạn khủng hoảng.

Đặt file tại: src/pipeline/subperiod_pipeline.py

Trả lời RQ3 — Chương 4.6:
  "Hướng lan truyền rủi ro có thay đổi qua COVID vs crypto crash không?"
  (Tích hợp AI: Neural Granger Causality cLSTM)

Giai đoạn phân tích:
  Pre-COVID       2018-01-01 → 2020-02-28
  COVID-19        2020-03-01 → 2021-06-30
  Post-COVID      2021-07-01 → 2022-04-30
  Crypto-Crash    2022-05-01 → 2022-12-31
  Banking-Crisis  2023-01-01 → 2023-06-30
  Recovery        2023-07-01 → 2024-12-31

Fixes (v2):
  - Import StandardScaler lên đầu file (không để trong hàm)
  - Fix DatetimeIndex: dùng boolean mask thay vì string slice
  - threshold tăng lên 0.3 để tránh full graph trong subperiod
"""

import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from sklearn.preprocessing import StandardScaler   # FIX: import lên đầu

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from config import (
    SUBPERIODS, NODE_NAMES,
    NOTEARS_LAMBDA1, NOTEARS_THRESHOLD,
    GRANGER_MAX_LAG,
    FIGURES_DIR, TABLES_DIR,
    BOOTSTRAP_N,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Class để tương thích format của Pipeline
# =============================================================================
class CausalResultWrapper:
    """
    Wrapper bọc kết quả ma trận của Neural Granger thành dạng chuẩn,
    cung cấp các hàm tính Hub/Sink/Degree tương thích với code vẽ biểu đồ.
    """
    def __init__(self, w_matrix: np.ndarray, node_names: List[str], runtime: float):
        self.adjacency_matrix = w_matrix
        self.binary_matrix    = (np.abs(w_matrix) > 0).astype(int)
        self.node_names       = node_names
        self.runtime_seconds  = runtime
        self.n_edges          = int(np.sum(self.binary_matrix))

    def out_degree(self) -> np.ndarray:
        return np.sum(self.binary_matrix, axis=1)

    def in_degree(self) -> np.ndarray:
        return np.sum(self.binary_matrix, axis=0)

    def hub_nodes(self, top_k=1) -> List[Tuple[str, int]]:
        out_d = self.out_degree()
        idx   = np.argsort(out_d)[::-1]
        return [(self.node_names[i], int(out_d[i])) for i in idx[:top_k] if out_d[i] > 0]

    def sink_nodes(self, top_k=1) -> List[Tuple[str, int]]:
        in_d = self.in_degree()
        idx  = np.argsort(in_d)[::-1]
        return [(self.node_names[i], int(in_d[i])) for i in idx[:top_k] if in_d[i] > 0]


# =============================================================================
# SubperiodPipeline
# =============================================================================
class SubperiodPipeline:
    """
    Chạy Causal Discovery (Neural Granger/NOTEARS) trên từng giai đoạn riêng biệt.
    So sánh cấu trúc đồ thị giữa các giai đoạn để tìm sự thay đổi dòng chảy rủi ro.

    Parameters
    ----------
    vol_df       : pd.DataFrame
        Rolling volatility đã xử lý. Index phải là DatetimeIndex hoặc date string.
    node_names   : list[str]
        Tên 5 thị trường.
    method       : str
        "neural" cho Neural Granger, "notears" cho NOTEARS.
    min_obs      : int
        Số quan sát tối thiểu để chạy một giai đoạn. Mặc định 60 (~3 tháng GD).
    """

    def __init__(
        self,
        vol_df    : pd.DataFrame,
        node_names: List[str],
        method    : str   = "neural",
        threshold : float = NOTEARS_THRESHOLD,
        lambda1   : float = NOTEARS_LAMBDA1,
        neural_lag: int   = GRANGER_MAX_LAG,
        min_obs   : int   = 60,
    ):
        # FIX: đảm bảo index là DatetimeIndex ngay khi khởi tạo
        if not isinstance(vol_df.index, pd.DatetimeIndex):
            vol_df = vol_df.copy()
            vol_df.index = pd.to_datetime(vol_df.index)

        self.vol_df     = vol_df
        self.node_names = node_names
        self.method     = method.lower()
        self.threshold  = threshold
        self.lambda1    = lambda1
        self.neural_lag = neural_lag
        self.min_obs    = min_obs

        # State
        self.period_results_: Dict[str, object]      = {}
        self.period_summary_: Optional[pd.DataFrame] = None

    # =========================================================================
    # run() — chạy toàn bộ subperiod analysis
    # =========================================================================

    def run(self) -> pd.DataFrame:
        """
        Chạy thuật toán cho tất cả giai đoạn trong SUBPERIODS.

        Returns
        -------
        pd.DataFrame  — bảng tóm tắt kết quả theo giai đoạn
        """
        logger.info("=" * 60)
        logger.info(f"SUBPERIOD ANALYSIS — ENGINE: {self.method.upper()}")
        logger.info("=" * 60)

        rows = []
        for period_name, (start, end) in SUBPERIODS.items():
            logger.info(f"\n[{period_name}] {start} → {end}")
            result = self._run_period_model(period_name, start, end)

            if result is None:
                rows.append(self._empty_row(period_name, start, end))
                continue

            self.period_results_[period_name] = result
            rows.append(self._summary_row(period_name, start, end, result))

        self.period_summary_ = pd.DataFrame(rows).set_index("Giai đoạn")
        _save_csv(self.period_summary_, TABLES_DIR / f"subperiod_results_{self.method}.csv")
        logger.info(f"\nBảng tóm tắt:\n{self.period_summary_.to_string()}")

        # Vẽ visuals
        self._plot_all()

        return self.period_summary_

    # =========================================================================
    # Chạy AI Model cho 1 giai đoạn
    # =========================================================================

    def _run_period_model(
        self, period_name: str, start: str, end: str
    ) -> Optional[object]:
        """
        Cắt dữ liệu theo giai đoạn → chuẩn hóa → chạy Model (Neural/NOTEARS).

        FIX: Dùng boolean mask thay vì string slice để tránh lỗi 0 quan sát
             khi index là DatetimeIndex.
        """
        # FIX: boolean mask — hoạt động đúng với mọi loại DatetimeIndex
        mask   = (self.vol_df.index >= pd.Timestamp(start)) & \
                 (self.vol_df.index <= pd.Timestamp(end))
        subset = self.vol_df.loc[mask].dropna()
        n_obs  = len(subset)

        if n_obs < self.min_obs:
            logger.warning(f"  Bỏ qua: chỉ có {n_obs} quan sát (< {self.min_obs})")
            return None

        # Chuẩn hóa StandardScaler cho từng subperiod riêng biệt
        X = StandardScaler().fit_transform(subset.values)

        if self.method == "neural":
            from src.models.neural_granger import NeuralGrangerCausality, NeuralGrangerConfig

            start_time = time.time()
            cfg = NeuralGrangerConfig(
                model_type   = "cLSTM",
                lag          = min(self.neural_lag, max(1, n_obs // 10)),
                hidden       = 16,
                lambda_group = 0.05,
                max_epochs   = 200,
                threshold    = 0.3,    # FIX: tăng từ 0.1 → 0.3 để tránh full graph
                verbose      = False,
            )
            model     = NeuralGrangerCausality(cfg)
            result_ng = model.fit(X, feature_names=self.node_names)
            w_matrix  = result_ng.adjacency_matrix
            runtime   = time.time() - start_time

            # Áp dụng threshold để tạo binary matrix
            binary = (w_matrix > cfg.threshold).astype(int)
            np.fill_diagonal(binary, 0)

            result = CausalResultWrapper.__new__(CausalResultWrapper)
            result.adjacency_matrix = w_matrix
            result.binary_matrix    = binary
            result.node_names       = self.node_names
            result.runtime_seconds  = runtime
            result.n_edges          = int(binary.sum())

        else:  # notears
            from src.models.notears import NOTEARS
            model  = NOTEARS(
                lambda1    = self.lambda1,
                threshold  = self.threshold,
                node_names = self.node_names,
            )
            result = model.fit_timed(X)

        logger.info(
            f"  n_obs={n_obs} | edges={result.n_edges} | "
            f"hub={result.hub_nodes(1)} | time={result.runtime_seconds:.2f}s"
        )
        return result

    # =========================================================================
    # Phân tích tổng hợp
    # =========================================================================

    def edge_frequency_matrix(self) -> pd.DataFrame:
        """
        Ma trận tần suất: freq[i,j] = tỷ lệ giai đoạn mà cạnh i→j tồn tại.
          freq = 1.0 → cạnh xuất hiện trong 100% giai đoạn (bền vững)
          freq = 0.0 → cạnh không bao giờ xuất hiện
        """
        if not self.period_results_:
            raise RuntimeError("Chạy run() trước")

        d    = len(self.node_names)
        freq = np.zeros((d, d))
        n    = len(self.period_results_)

        for result in self.period_results_.values():
            freq += result.binary_matrix

        freq /= n
        return pd.DataFrame(
            freq.round(3),
            index   = self.node_names,
            columns = self.node_names,
        )

    def stable_edges(self, min_freq: float = 0.5) -> List[dict]:
        """Trả về danh sách cạnh ổn định (tần suất ≥ min_freq)."""
        freq_df = self.edge_frequency_matrix()
        edges   = []
        for i, src in enumerate(self.node_names):
            for j, tgt in enumerate(self.node_names):
                f = float(freq_df.iloc[i, j])
                if f >= min_freq:
                    edges.append({
                        "source"   : src,
                        "target"   : tgt,
                        "frequency": round(f, 3),
                        "stability": "Rất ổn định" if f >= 0.8 else "Ổn định",
                    })
        return sorted(edges, key=lambda x: -x["frequency"])

    def crisis_comparison(self) -> pd.DataFrame:
        """So sánh cấu trúc đồ thị giữa các giai đoạn bằng SHD."""
        from src.evaluation.metrics import structural_hamming_distance

        periods = list(self.period_results_.keys())
        n       = len(periods)
        shd_mat = np.zeros((n, n), dtype=int)

        for i in range(n):
            for j in range(n):
                if i != j:
                    shd_mat[i, j] = structural_hamming_distance(
                        self.period_results_[periods[i]].binary_matrix,
                        self.period_results_[periods[j]].binary_matrix,
                    )

        df = pd.DataFrame(shd_mat, index=periods, columns=periods)
        _save_csv(df, TABLES_DIR / f"subperiod_shd_matrix_{self.method}.csv")
        return df

    def hub_evolution(self) -> pd.DataFrame:
        """Theo dõi sự thay đổi hub node qua các giai đoạn."""
        rows = []
        for period_name, result in self.period_results_.items():
            row = {"Giai đoạn": period_name}
            out = result.out_degree()
            for i, name in enumerate(self.node_names):
                row[name] = int(out[i])
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Giai đoạn")
        _save_csv(df, TABLES_DIR / f"hub_evolution_{self.method}.csv")
        return df

    # =========================================================================
    # Visualization
    # =========================================================================

    def _plot_all(self) -> None:
        """Vẽ tất cả visuals subperiod."""
        if not self.period_results_:
            return

        # 1. Edge stability heatmap
        try:
            from src.visualization.risk_map import plot_edge_stability
            freq_df = self.edge_frequency_matrix()
            plot_edge_stability(
                freq_matrix = freq_df.values,
                node_names  = self.node_names,
                output_path = str(FIGURES_DIR / f"edge_stability_{self.method}.png"),
            )
        except Exception as e:
            logger.warning(f"Không vẽ được edge stability: {e}")

        # 2. DAG cho từng giai đoạn
        try:
            from src.visualization.causal_graph import plot_dag
            for period_name, result in self.period_results_.items():
                safe_name = period_name.replace(" ", "_").replace("-", "").lower()
                plot_dag(
                    binary_matrix = result.binary_matrix,
                    node_names    = self.node_names,
                    title         = f"{self.method.upper()} DAG — {period_name}",
                    output_path   = str(FIGURES_DIR / f"dag_subperiod_{self.method}_{safe_name}.png"),
                    show_weights  = False,
                )
        except Exception as e:
            logger.warning(f"Không vẽ được subperiod DAGs: {e}")

        # 3. Hub evolution line chart
        try:
            self._plot_hub_evolution()
        except Exception as e:
            logger.warning(f"Không vẽ được hub evolution: {e}")

    def _plot_hub_evolution(self) -> None:
        """Line chart theo dõi out-degree của từng thị trường qua các giai đoạn."""
        import matplotlib.pyplot as plt
        from config import MARKET_COLORS as _C, MARKET_LABELS as _L, FIG_DPI

        hub_df = self.hub_evolution()
        fig, ax = plt.subplots(figsize=(11, 5))

        for market in self.node_names:
            if market not in hub_df.columns:
                continue
            ax.plot(
                hub_df.index, hub_df[market],
                marker    = "o",
                linewidth = 2,
                markersize= 7,
                color     = _C.get(market, "#aaa"),
                label     = _L.get(market, market),
            )

        ax.set_xlabel("Giai đoạn", fontsize=11)
        ax.set_ylabel("Out-degree (Sức lan tỏa)", fontsize=11)
        ax.set_title(
            f"Sự thay đổi Hub Node qua các Giai đoạn ({self.method.upper()})\n"
            "Thị trường nào là nguồn phát rủi ro mạnh nhất?",
            fontsize   = 12,
            fontweight = "bold",
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(axis="x", rotation=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        out = str(FIGURES_DIR / f"hub_evolution_{self.method}.png")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
        logger.info(f"Đã lưu: {out}")

    # =========================================================================
    # Summary & Print
    # =========================================================================

    def print_summary(self) -> None:
        """In tóm tắt kết quả subperiod ra console."""
        if self.period_summary_ is None:
            logger.warning("Chưa chạy run()")
            return

        print("\n" + "═" * 70)
        print(f"  SUBPERIOD ANALYSIS — Động lực Nhân quả ({self.method.upper()})")
        print("═" * 70)
        print(self.period_summary_.to_string())

        # Stable edges
        stable = self.stable_edges(min_freq=0.5)
        print(f"\n  Cạnh rủi ro cốt lõi (tần suất ≥ 50%): {len(stable)} cạnh")
        for e in stable:
            print(f"    {e['source']:10s} → {e['target']:10s} | freq={e['frequency']} | {e['stability']}")

        # Crisis comparison
        try:
            shd_df = self.crisis_comparison()
            print(f"\n  SHD (Mức độ biến động cấu trúc) giữa các giai đoạn:\n{shd_df.to_string()}")
        except Exception:
            pass

        print("═" * 70 + "\n")

    # =========================================================================
    # Private static helpers
    # =========================================================================

    @staticmethod
    def _summary_row(
        period_name: str, start: str, end: str, result: object
    ) -> dict:
        d        = result.binary_matrix.shape[0]
        density  = result.n_edges / (d * (d - 1)) if d > 1 else 0
        hub      = result.hub_nodes(top_k=1)
        hub_str  = f"{hub[0][0]} (out={hub[0][1]})" if hub else "-"
        sink     = result.sink_nodes(top_k=1)
        sink_str = f"{sink[0][0]} (in={sink[0][1]})" if sink else "-"

        return {
            "Giai đoạn"          : period_name,
            "Thời gian"          : f"{start[:7]} ~ {end[:7]}",
            "Số cạnh"            : result.n_edges,
            "Graph Density"      : round(density, 3),
            "Hub (phát rủi ro)"  : hub_str,
            "Sink (nhận rủi ro)" : sink_str,
            "Runtime (s)"        : round(result.runtime_seconds, 2),
        }

    @staticmethod
    def _empty_row(period_name: str, start: str, end: str) -> dict:
        return {
            "Giai đoạn"          : period_name,
            "Thời gian"          : f"{start[:7]} ~ {end[:7]}",
            "Số cạnh"            : "-",
            "Graph Density"      : "-",
            "Hub (phát rủi ro)"  : "N/A (ít dữ liệu)",
            "Sink (nhận rủi ro)" : "N/A",
            "Runtime (s)"        : "-",
        }


# =============================================================================
# Module-level helpers
# =============================================================================

def _save_csv(df: pd.DataFrame, path, index: bool = True) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info(f"Đã lưu: {path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    """
    Chạy subperiod analysis độc lập (sau khi đã có vol_df từ FullPipeline).

    Usage:
        python -m src.pipeline.subperiod_pipeline
    """
    from src.utils.logger import setup_logger
    setup_logger("KLTN")

    vol_path = Path("reports/tables/volatility.csv")
    if not vol_path.exists():
        logger.error(
            "Chưa có file volatility.csv. "
            "Chạy FullPipeline trước hoặc cung cấp vol_df trực tiếp."
        )
        sys.exit(1)

    vol_df     = pd.read_csv(vol_path, index_col=0, parse_dates=True)
    node_names = list(vol_df.columns)

    subpipe = SubperiodPipeline(
        vol_df     = vol_df,
        node_names = node_names,
        method     = "neural",
        threshold  = 0.3,
    )
    subpipe.run()
    subpipe.print_summary()