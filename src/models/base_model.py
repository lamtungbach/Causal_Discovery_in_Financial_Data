"""
src/models/base_model.py
========================
Abstract base class chuẩn hóa interface cho các causal discovery methods.

Methods được hỗ trợ:
    - NeuralGranger  (cMLP / cLSTM)  ← thuật toán chính
    - NOTEARS                         ← baseline
    - PC Algorithm                    ← baseline

Thay đổi so với phiên bản cũ:
    - Thêm ModelType enum để định danh method
    - CausalResult bổ sung trường: model_type, neural_granger_result
      (lưu NeuralGrangerResult gốc để dùng get_risk_scores / get_causal_edges)
    - Thêm risk_scores(), causal_edges() để pipeline/EWS truy xuất trực tiếp
    - sink_nodes(), degree_table() giữ nguyên
    - n_edges vẫn tự tính từ binary_matrix
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from src.models.neural_granger import NeuralGrangerResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum: định danh method
# ---------------------------------------------------------------------------

class ModelType(str, Enum):
    NEURAL_GRANGER_CMLP  = "NeuralGranger-cMLP"
    NEURAL_GRANGER_CLSTM = "NeuralGranger-cLSTM"
    NOTEARS              = "NOTEARS"
    PC                   = "PC"
    UNKNOWN              = "Unknown"


# ---------------------------------------------------------------------------
# CausalResult
# ---------------------------------------------------------------------------

@dataclass
class CausalResult:
    """
    Kết quả chuẩn hóa từ bất kỳ causal discovery model nào.
    Dùng để so sánh thống nhất giữa NeuralGranger, NOTEARS, PC.
    """

    method_name      : str
    adjacency_matrix : np.ndarray          # Ma trận W (có trọng số hoặc normalised)
    binary_matrix    : np.ndarray          # Ma trận 0/1 sau threshold
    runtime_seconds  : float               # Được gán bởi fit_timed()

    # ── Metadata ──────────────────────────────────────────────────────────
    model_type       : ModelType = ModelType.UNKNOWN
    n_edges          : int       = 0       # Tự tính trong __post_init__
    metadata         : dict      = field(default_factory=dict)

    # ── Neural Granger specific (None với NOTEARS / PC) ───────────────────
    neural_granger_result: Optional["NeuralGrangerResult"] = field(
        default=None, repr=False
    )

    # ──────────────────────────────────────────────────────────────────────
    def __post_init__(self):
        # Luôn tính lại từ binary_matrix
        self.n_edges = int(self.binary_matrix.sum())

    # ── Node helpers ──────────────────────────────────────────────────────

    @property
    def node_names(self) -> List[str]:
        return self.metadata.get(
            "node_names",
            [f"X{i}" for i in range(self.binary_matrix.shape[0])],
        )

    def out_degree(self) -> np.ndarray:
        """Số cạnh đi ra — đo lường khả năng PHÁT rủi ro."""
        return self.binary_matrix.sum(axis=1)

    def in_degree(self) -> np.ndarray:
        """Số cạnh đi vào — đo lường khả năng NHẬN rủi ro."""
        return self.binary_matrix.sum(axis=0)

    def hub_nodes(self, top_k: int = 2) -> List[Tuple[str, int]]:
        """Top-k node phát rủi ro mạnh nhất (out-degree cao nhất)."""
        names   = self.node_names
        out_deg = self.out_degree()
        idx     = np.argsort(out_deg)[::-1][:top_k]
        return [(names[i], int(out_deg[i])) for i in idx]

    def sink_nodes(self, top_k: int = 2) -> List[Tuple[str, int]]:
        """Top-k node nhận rủi ro nhiều nhất (in-degree cao nhất)."""
        names  = self.node_names
        in_deg = self.in_degree()
        idx    = np.argsort(in_deg)[::-1][:top_k]
        return [(names[i], int(in_deg[i])) for i in idx]

    def edge_list(self) -> List[dict]:
        """
        Danh sách cạnh dạng dict — tiện cho export JSON / visualization.

        Với NeuralGranger: dùng get_causal_edges() để có thêm trường 'cause'.
        """
        names = self.node_names
        edges = []
        for i in range(self.binary_matrix.shape[0]):
            for j in range(self.binary_matrix.shape[1]):
                if self.binary_matrix[i, j] == 1:
                    edges.append({
                        "source": names[i],
                        "target": names[j],
                        "weight": round(float(self.adjacency_matrix[i, j]), 4),
                    })
        return edges

    def degree_table(self) -> "pd.DataFrame":
        """Bảng in/out-degree + role cho từng thị trường."""
        import pandas as pd
        names = self.node_names
        out   = self.out_degree().astype(int)
        ins   = self.in_degree().astype(int)
        return pd.DataFrame(
            {
                "Market"    : names,
                "Out-degree": out,
                "In-degree" : ins,
                "Net"       : (out - ins),
                "Role"      : [
                    "Source"  if o > i else
                    ("Sink"   if i > o else "Neutral")
                    for o, i in zip(out, ins)
                ],
            }
        ).set_index("Market")

    # ── Neural Granger specific helpers ───────────────────────────────────

    def risk_scores(self) -> Optional[Dict[str, float]]:
        """
        Risk Score per variable dựa trên out-degree strength của Neural Granger.

        - Nếu result đến từ NeuralGranger: dùng adjacency thực (continuous).
        - Fallback: tính từ out_degree() (integer) cho NOTEARS / PC.

        Returns dict {name: score} hoặc None nếu chưa fit.
        """
        if self.neural_granger_result is not None:
            # Dùng continuous adjacency → score chính xác hơn
            A = self.neural_granger_result.adjacency_matrix
            names = self.neural_granger_result.feature_names
            return {name: float(A[i].sum()) for i, name in enumerate(names)}

        # Fallback cho NOTEARS / PC
        names = self.node_names
        out   = self.out_degree()
        total = float(out.sum()) or 1.0
        return {name: float(out[i]) / total for i, name in enumerate(names)}

    def causal_edges(self) -> List[Tuple[str, str, float]]:
        """
        Danh sách (cause, effect, strength) của tất cả cạnh nhân quả.

        - NeuralGranger: dùng get_causal_edges() gốc (đã sort theo strength).
        - NOTEARS / PC : xây từ edge_list().

        Returns list of (cause, effect, strength).
        """
        if self.neural_granger_result is not None:
            return self.neural_granger_result.get_causal_edges() if hasattr(
                self.neural_granger_result, "get_causal_edges"
            ) else []

        # Fallback
        return [
            (e["source"], e["target"], e["weight"])
            for e in sorted(self.edge_list(), key=lambda x: -x["weight"])
        ]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseCausalModel(ABC):
    """Abstract base class cho tất cả causal discovery methods."""

    def __init__(self, node_names: Optional[List[str]] = None):
        self.node_names = node_names
        self._result: Optional[CausalResult] = None

    # ──────────────────────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, X: np.ndarray) -> CausalResult:
        """
        Học cấu trúc nhân quả từ dữ liệu X.

        Parameters
        ----------
        X : np.ndarray  shape (n_samples, n_variables)
            Ví dụ: rolling volatility annualised của 5 tài sản.

        Returns
        -------
        CausalResult
        """

    # ──────────────────────────────────────────────────────────────────────

    def fit_timed(self, X: np.ndarray) -> CausalResult:
        """fit() kèm tự động đo runtime và ghi log."""
        logger.info(f"[{self.__class__.__name__}] Bắt đầu | X={X.shape}")
        t0                     = time.perf_counter()
        result                 = self.fit(X)
        result.runtime_seconds = time.perf_counter() - t0
        self._result           = result
        logger.info(
            f"[{self.__class__.__name__}] Xong | "
            f"method={result.model_type.value} | "
            f"edges={result.n_edges} | "
            f"time={result.runtime_seconds:.2f}s"
        )
        return result

    # ──────────────────────────────────────────────────────────────────────

    @property
    def result(self) -> Optional[CausalResult]:
        return self._result

    def summary(self) -> str:
        if self._result is None:
            return f"{self.__class__.__name__}: chưa fit"
        r = self._result
        lines = [
            f"=== {r.method_name} ===",
            f"  Model type : {r.model_type.value}",
            f"  Edges      : {r.n_edges}",
            f"  Runtime    : {r.runtime_seconds:.3f}s",
            f"  Hub nodes  : {r.hub_nodes()}",
            f"  Sink nodes : {r.sink_nodes()}",
        ]
        scores = r.risk_scores()
        if scores:
            lines.append("  Risk scores:")
            for name, score in sorted(scores.items(), key=lambda x: -x[1]):
                lines.append(f"    {name:<12}: {score:.4f}")
        return "\n".join(lines)