"""
NOTEARS Model - Nâng cấp với threshold sensitivity analysis.
Zheng et al. (2018) - DAGs with NO TEARS.
"""

import numpy as np
import logging
from typing import Optional, List, Tuple
from .base_model import BaseCausalModel, CausalResult

logger = logging.getLogger(__name__)


class NOTEARS(BaseCausalModel):
    """
    NOTEARS: Continuous optimization cho structure learning.

    Bài toán tối ưu:
        min F(W) = (1/2n)||X - XW||²_F
        subject to h(W) = tr(e^(W∘W)) - d = 0

    Parameters
    ----------
    lambda1 : float
        L1 regularization (sparsity). Mặc định 0.1.
    threshold : float
        Ngưỡng cắt cạnh yếu sau khi tối ưu. Mặc định 0.2.
    max_iter : int
        Số vòng lặp Augmented Lagrangian tối đa.
    """

    def __init__(
        self,
        lambda1: float = 0.1,
        threshold: float = 0.2,
        max_iter: int = 100,
        node_names: Optional[List[str]] = None,
    ):
        super().__init__(node_names=node_names)
        self.lambda1 = lambda1
        self.threshold = threshold
        self.max_iter = max_iter
        self.W_est_raw: Optional[np.ndarray] = None  # Ma trận trước threshold

    # ------------------------------------------------------------------
    # Core optimization
    # ------------------------------------------------------------------

    @staticmethod
    def _h(W: np.ndarray) -> float:
        """Acyclicity constraint: h(W) = tr(e^(W∘W)) - d."""
        d = W.shape[0]
        return float(np.trace(np.linalg.matrix_power(
            np.eye(d) + W * W / d, d
        )) - d)

    @staticmethod
    def _h_grad(W: np.ndarray) -> np.ndarray:
        """Gradient của h(W) theo W."""
        d = W.shape[0]
        E = np.linalg.matrix_power(np.eye(d) + W * W / d, d - 1)
        return (2 / d) * W * E.T

    def _squared_loss(self, X: np.ndarray, W: np.ndarray) -> Tuple[float, np.ndarray]:
        """Loss + gradient: (1/2n)||X - XW||²_F."""
        n = X.shape[0]
        M = X @ W
        R = X - M
        loss = 0.5 / n * (R ** 2).sum()
        grad = -1.0 / n * X.T @ R
        return loss, grad

    def _fit_notears(self, X: np.ndarray) -> np.ndarray:
        """
        Augmented Lagrangian method để học W.
        Dùng numpy thuần — không cần thư viện ngoài.
        """
        n, d = X.shape
        W = np.zeros((d, d))

        # Augmented Lagrangian params
        rho, alpha, h_prev = 1.0, 0.0, np.inf
        rho_max = 1e16
        h_tol = 1e-8

        for iteration in range(self.max_iter):
            # --- Inner loop: gradient descent ---
            for _ in range(300):
                loss, grad_loss = self._squared_loss(X, W)
                h_val = self._h(W)
                grad_h = self._h_grad(W)

                # Augmented Lagrangian gradient
                obj_grad = grad_loss + (alpha + rho * h_val) * grad_h

                # Soft-thresholding (L1 proximal)
                lr = 1e-3
                W_new = W - lr * obj_grad
                # L1 proximal
                W_new = np.sign(W_new) * np.maximum(
                    np.abs(W_new) - self.lambda1 * lr, 0
                )
                # Zero diagonal (no self-loops)
                np.fill_diagonal(W_new, 0)

                if np.max(np.abs(W_new - W)) < 1e-6:
                    W = W_new
                    break
                W = W_new

            h_val = self._h(W)
            logger.debug(f"  Iter {iteration}: h={h_val:.4e}, rho={rho:.2e}")

            # --- Outer loop: update dual ---
            if h_val > 0.25 * h_prev:
                rho = min(rho * 10, rho_max)
            alpha += rho * h_val
            h_prev = h_val

            if h_val <= h_tol:
                logger.info(f"  Hội tụ tại iteration {iteration}, h={h_val:.2e}")
                break

        np.fill_diagonal(W, 0)
        return W

    # ------------------------------------------------------------------
    # BaseCausalModel interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> CausalResult:
        n, d = X.shape
        node_names = self.node_names or [f"X{i}" for i in range(d)]

        W_raw = self._fit_notears(X)
        self.W_est_raw = W_raw

        W_binary = (np.abs(W_raw) > self.threshold).astype(int)
        np.fill_diagonal(W_binary, 0)

        return CausalResult(
            method_name="NOTEARS",
            adjacency_matrix=W_raw,
            binary_matrix=W_binary,
            runtime_seconds=0.0,  # sẽ được set bởi fit_timed
            n_edges=int(W_binary.sum()),
            metadata={
                "node_names": node_names,
                "lambda1": self.lambda1,
                "threshold": self.threshold,
            },
        )

    # ------------------------------------------------------------------
    # Threshold sensitivity analysis (quan trọng cho khóa luận!)
    # ------------------------------------------------------------------

    def threshold_sensitivity(
        self,
        thresholds: Optional[List[float]] = None,
    ) -> List[dict]:
        """
        Phân tích độ nhạy của kết quả theo ngưỡng cắt cạnh.

        Returns
        -------
        List[dict]: Mỗi dict chứa threshold, n_edges, density
        """
        if self.W_est_raw is None:
            raise RuntimeError("Phải fit() trước khi gọi threshold_sensitivity()")

        thresholds = thresholds or [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        d = self.W_est_raw.shape[0]
        results = []

        for thr in thresholds:
            W_bin = (np.abs(self.W_est_raw) > thr).astype(int)
            np.fill_diagonal(W_bin, 0)
            n_edges = int(W_bin.sum())
            density = n_edges / (d * (d - 1))  # max edges = d*(d-1)
            results.append({
                "threshold": thr,
                "n_edges": n_edges,
                "density": round(density, 4),
            })

        return results

    def get_edge_weights(self, node_names: Optional[List[str]] = None) -> List[dict]:
        """Trả về danh sách cạnh với trọng số để vẽ đồ thị."""
        if self.W_est_raw is None:
            raise RuntimeError("Phải fit() trước")

        names = node_names or self.node_names or [
            f"X{i}" for i in range(self.W_est_raw.shape[0])
        ]
        W_bin = (np.abs(self.W_est_raw) > self.threshold).astype(int)
        np.fill_diagonal(W_bin, 0)

        edges = []
        for i in range(W_bin.shape[0]):
            for j in range(W_bin.shape[1]):
                if W_bin[i, j] == 1:
                    edges.append({
                        "source": names[i],
                        "target": names[j],
                        "weight": round(float(self.W_est_raw[i, j]), 4),
                    })
        return edges
