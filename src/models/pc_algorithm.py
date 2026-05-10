"""
PC Algorithm - Constraint-based Causal Discovery.
Kế thừa BaseCausalModel.
"""

import numpy as np
import logging
from typing import Optional, List
from .base_model import BaseCausalModel, CausalResult

logger = logging.getLogger(__name__)


class PCAlgorithm(BaseCausalModel):
    """
    PC Algorithm: học DAG từ conditional independence tests.

    Steps:
    1. Skeleton discovery: loại bỏ cạnh dựa trên CI test
    2. V-structure orientation: X → Z ← Y nếu Z không trong sep(X,Y)
    3. Meek rules: định hướng các cạnh còn lại

    Parameters
    ----------
    alpha : float
        Ngưỡng significant cho CI test. Mặc định 0.05.
    ci_test : str
        Loại CI test: 'fisherz' | 'kci'. Mặc định 'fisherz'.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        ci_test: str = "fisherz",
        node_names: Optional[List[str]] = None,
    ):
        super().__init__(node_names=node_names)
        self.alpha = alpha
        self.ci_test = ci_test
        self.cpdag: Optional[np.ndarray] = None  # Raw CPDAG output

    def fit(self, X: np.ndarray) -> CausalResult:
        try:
            from causallearn.search.ConstraintBased.PC import pc
            from causallearn.utils.cit import fisherz, kci
        except ImportError:
            raise ImportError("Cần cài: pip install causal-learn")

        n, d = X.shape
        node_names = self.node_names or [f"X{i}" for i in range(d)]

        logger.info(f"Chạy PC Algorithm: alpha={self.alpha}, CI test={self.ci_test}")

        # Chạy PC
        cg = pc(
            data=X,
            alpha=self.alpha,
            indep_test=self.ci_test,
            stable=True,        # Stable PC (ít phụ thuộc thứ tự biến)
            uc_rule=0,          # Kimdel UC rule
            uc_priority=-1,
            verbose=False,
        )

        # Lấy adjacency matrix từ CPDAG
        # causal-learn dùng convention: G.graph[i,j]=1 & G.graph[j,i]=-1 → i→j
        cpdag = cg.G.graph  # shape (d, d)
        self.cpdag = cpdag

        # Chuyển sang binary directed matrix
        binary_matrix = np.zeros((d, d), dtype=int)
        for i in range(d):
            for j in range(d):
                # i → j: cpdag[j,i]=1 AND cpdag[i,j]=-1
                if cpdag[j, i] == 1 and cpdag[i, j] == -1:
                    binary_matrix[i, j] = 1
                # Undirected edge: cpdag[i,j]=1 AND cpdag[j,i]=1
                # → đưa vào cả hai hướng để không mất thông tin
                elif cpdag[i, j] == 1 and cpdag[j, i] == 1:
                    binary_matrix[i, j] = 1  # ghi nhận có liên hệ

        return CausalResult(
            method_name="PC Algorithm",
            adjacency_matrix=cpdag.astype(float),
            binary_matrix=binary_matrix,
            runtime_seconds=0.0,
            n_edges=int(binary_matrix.sum()),
            metadata={
                "node_names": node_names,
                "alpha": self.alpha,
                "ci_test": self.ci_test,
                "n_undirected": self._count_undirected(),
            },
        )

    def _count_undirected(self) -> int:
        """Đếm số cạnh chưa xác định hướng trong CPDAG."""
        if self.cpdag is None:
            return 0
        count = 0
        d = self.cpdag.shape[0]
        for i in range(d):
            for j in range(i + 1, d):
                if self.cpdag[i, j] == 1 and self.cpdag[j, i] == 1:
                    count += 1
        return count

    def orientation_summary(self) -> dict:
        """Tóm tắt số cạnh directed vs undirected — hữu ích cho Chương 4."""
        if self._result is None:
            raise RuntimeError("Phải fit() trước")
        n_undirected = self._count_undirected()
        n_directed = self._result.n_edges - n_undirected
        return {
            "total_edges": self._result.n_edges,
            "directed": n_directed,
            "undirected": n_undirected,
            "orientation_rate": round(
                n_directed / max(self._result.n_edges, 1), 3
            ),
        }
