"""
neural_granger.py
=================
Neural Granger Causality using cMLP and cLSTM architectures.

Based on:
    Tank et al. (2021) "Neural Granger Causality"
    IEEE Transactions on Pattern Analysis and Machine Intelligence.

Supports:
    - cMLP  : component-wise MLP with shared lag structure
    - cLSTM : component-wise LSTM with shared lag structure

Both models learn a sparse adjacency matrix W that encodes Granger-causal
relationships: W[i, j] != 0  ⟹  time series j Granger-causes time series i.

Regularisation:
    - Group LASSO on input weights (one group per predictor)  → promotes
      exact zeros and gives a hard adjacency matrix.
    - Optional Ridge on remaining weights for stability.

Fixes (v2):
    - get_adjacency() dùng global max thay vì row max → tránh full graph
    - fill_diagonal(0) TRƯỚC khi normalize → self-loop không ảnh hưởng scale
    - lambda_group mặc định tăng lên 0.05
    - threshold mặc định 0.15
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class NeuralGrangerResult:
    """Output returned by NeuralGrangerCausality.fit()."""

    adjacency_matrix: np.ndarray          # (p, p)  float  W[i,j] in [0,1]
    causal_matrix: np.ndarray             # (p, p)  bool   thresholded
    weights_history: List[np.ndarray]     # adjacency per recorded epoch
    loss_history: Dict[str, List[float]]  # {'total', 'mse', 'reg'}
    feature_names: List[str]
    model_type: str                        # 'cMLP' | 'cLSTM'
    threshold: float
    n_epochs_trained: int

    # -----------------------------------------------------------------------
    def summary(self) -> str:
        p = len(self.feature_names)
        edges = int(self.causal_matrix.sum())
        lines = [
            f"Neural Granger Causality  [{self.model_type}]",
            f"  Variables : {self.feature_names}",
            f"  Threshold : {self.threshold}",
            f"  Edges found : {edges} / {p * (p - 1)}",
            "",
            "Adjacency matrix (strength):",
        ]
        header = "       " + "  ".join(f"{n[:6]:>6}" for n in self.feature_names)
        lines.append(header)
        for i, name in enumerate(self.feature_names):
            row = f"{name[:6]:>6} " + "  ".join(
                f"{self.adjacency_matrix[i, j]:6.3f}" for j in range(p)
            )
            lines.append(row)
        lines.append("")
        lines.append("Causal graph (boolean):")
        lines.append(header)
        for i, name in enumerate(self.feature_names):
            row = f"{name[:6]:>6} " + "  ".join(
                f"{'1' if self.causal_matrix[i, j] else '0':>6}" for j in range(p)
            )
            lines.append(row)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# cMLP  (component-wise MLP)
# ---------------------------------------------------------------------------

class _ComponentMLP(nn.Module):
    """
    One MLP per target variable i.  Input is x_{-i} (all lags of all series).
    The first linear layer has shape (hidden, p * lag) and its weights are
    used for group-LASSO regularisation (group = predictor j, size = lag).
    """

    def __init__(self, p: int, lag: int, hidden: int, n_layers: int):
        super().__init__()
        self.p = p
        self.lag = lag
        input_dim = p * lag

        layers: List[nn.Module] = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    @property
    def input_weights(self) -> torch.Tensor:
        """First layer weights  shape (hidden, p * lag)."""
        return self.net[0].weight  # type: ignore[index]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, p * lag)  →  (batch, 1)"""
        return self.net(x)


class cMLP(nn.Module):
    """
    Component-wise MLP for Neural Granger Causality.
    Maintains p separate MLPs, one per target variable.
    """

    def __init__(self, p: int, lag: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.p = p
        self.lag = lag
        self.models = nn.ModuleList(
            [_ComponentMLP(p, lag, hidden, n_layers) for _ in range(p)]
        )

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, lag, p)
        returns : (batch, p)  — one-step-ahead predictions
        """
        batch = x.shape[0]
        # reorder to (batch, p * lag)  — group by predictor
        x_grouped = x.permute(0, 2, 1).reshape(batch, -1)  # (batch, p * lag)
        preds = [m(x_grouped) for m in self.models]         # list of (batch,1)
        return torch.cat(preds, dim=1)                        # (batch, p)

    # -----------------------------------------------------------------------
    def group_lasso_penalty(self) -> torch.Tensor:
        """
        Sum of L2 norms over input-weight groups.
        Group j of model i = columns [j*lag : (j+1)*lag] of the first layer.
        """
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for model in self.models:
            W = model.input_weights          # (hidden, p * lag)
            for j in range(self.p):
                g = W[:, j * self.lag: (j + 1) * self.lag]  # (hidden, lag)
                penalty = penalty + g.norm()
        return penalty

    # -----------------------------------------------------------------------
    def get_adjacency(self) -> np.ndarray:
        """
        Returns (p, p) matrix where A[i, j] = L2 norm of the input group of
        predictor j in the MLP for target i.

        FIX: Normalize theo GLOBAL max (không phải row max) để tránh full graph.
        Self-loop được zero-out TRƯỚC khi normalize.
        """
        A = np.zeros((self.p, self.p))
        for i, model in enumerate(self.models):
            W = model.input_weights.detach().cpu()
            for j in range(self.p):
                g = W[:, j * self.lag: (j + 1) * self.lag]
                A[i, j] = float(g.norm())

        # Zero-out self-loops trước normalize
        np.fill_diagonal(A, 0.0)

        # Normalize theo GLOBAL max → giá trị nhỏ vẫn nhỏ sau normalize
        global_max = A.max()
        if global_max > 0:
            A = A / global_max

        return A


# ---------------------------------------------------------------------------
# cLSTM  (component-wise LSTM)
# ---------------------------------------------------------------------------

class _ComponentLSTM(nn.Module):
    """
    One LSTM per target variable i.
    Input at each step: x_t  shape (p,).
    The input-to-hidden weights (W_i, W_f, W_g, W_o part of ih_l0, shape
    (4*hidden, p)) are used for group-LASSO.
    """

    def __init__(self, p: int, hidden: int):
        super().__init__()
        self.p = p
        self.hidden = hidden
        self.lstm = nn.LSTM(input_size=p, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    @property
    def input_weights(self) -> torch.Tensor:
        """ih weights  shape (4*hidden, p)."""
        return self.lstm.weight_ih_l0  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, lag, p)  →  (batch, 1)"""
        out, _ = self.lstm(x)           # (batch, lag, hidden)
        last = out[:, -1, :]            # (batch, hidden)
        return self.fc(last)            # (batch, 1)


class cLSTM(nn.Module):
    """
    Component-wise LSTM for Neural Granger Causality.
    """

    def __init__(self, p: int, lag: int, hidden: int = 64):
        super().__init__()
        self.p = p
        self.lag = lag
        self.models = nn.ModuleList(
            [_ComponentLSTM(p, hidden) for _ in range(p)]
        )

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, lag, p)  →  (batch, p)"""
        preds = [m(x) for m in self.models]   # list of (batch, 1)
        return torch.cat(preds, dim=1)          # (batch, p)

    # -----------------------------------------------------------------------
    def group_lasso_penalty(self) -> torch.Tensor:
        """
        Group-LASSO over input-to-hidden weights.
        Group j = column j (across all 4 gates) of ih weights.
        shape of ih: (4*hidden, p)  →  group j = ih[:, j]  shape (4*hidden,)
        """
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for model in self.models:
            W = model.input_weights          # (4*hidden, p)
            for j in range(self.p):
                g = W[:, j]                  # (4*hidden,)
                penalty = penalty + g.norm()
        return penalty

    # -----------------------------------------------------------------------
    def get_adjacency(self) -> np.ndarray:
        """
        Returns (p, p) normalised adjacency matrix.

        FIX: Normalize theo GLOBAL max (không phải row max) để tránh full graph.
        Self-loop được zero-out TRƯỚC khi normalize.
        """
        A = np.zeros((self.p, self.p))
        for i, model in enumerate(self.models):
            W = model.input_weights.detach().cpu()   # (4*hidden, p)
            for j in range(self.p):
                A[i, j] = float(W[:, j].norm())

        # Zero-out self-loops trước normalize
        np.fill_diagonal(A, 0.0)

        # Normalize theo GLOBAL max
        global_max = A.max()
        if global_max > 0:
            A = A / global_max

        return A


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

@dataclass
class NeuralGrangerConfig:
    """Hyper-parameters for NeuralGrangerCausality."""

    model_type: Literal["cMLP", "cLSTM"] = "cMLP"
    lag: int = 5                    # number of lags fed as input
    hidden: int = 64                # hidden units per component network
    n_layers: int = 2               # depth  (cMLP only; cLSTM uses 1 layer)
    lambda_group: float = 0.05      # group-LASSO weight — FIX: tăng từ 0.01 → 0.05
    lambda_ridge: float = 1e-4      # ridge on all other parameters
    lr: float = 1e-3
    max_epochs: int = 500
    patience: int = 30              # early stopping on val loss
    batch_size: int = 64
    val_fraction: float = 0.15
    threshold: float = 0.2         # FIX: tăng từ 0.1 → 0.15 (phù hợp global-max norm)
    device: str = "cpu"
    seed: int = 42
    record_every: int = 50          # save adjacency snapshot every N epochs
    verbose: bool = True


class NeuralGrangerCausality:
    """
    Neural Granger Causality estimator.

    Learns directed causal structure from multivariate time-series data by
    training component-wise neural networks (cMLP or cLSTM) with group-LASSO
    regularisation that promotes exact sparsity in the input weights.

    Usage
    -----
    >>> ngc = NeuralGrangerCausality(config)
    >>> result = ngc.fit(data, feature_names=["SP500","VNIndex","Gold","Oil","BTC"])
    >>> print(result.summary())
    """

    def __init__(self, config: Optional[NeuralGrangerConfig] = None):
        self.config = config or NeuralGrangerConfig()
        self._model: Optional[nn.Module] = None
        self._result: Optional[NeuralGrangerResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> NeuralGrangerResult:
        """
        Parameters
        ----------
        data : np.ndarray  shape (T, p)
            Multivariate time series (e.g. annualised rolling volatility).
        feature_names : list[str], optional
            Names for each variable.  Defaults to ['X0', 'X1', ...].

        Returns
        -------
        NeuralGrangerResult
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        T, p = data.shape
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(p)]

        device = torch.device(cfg.device)

        # ---- build model ------------------------------------------------
        if cfg.model_type == "cMLP":
            model = cMLP(p, cfg.lag, cfg.hidden, cfg.n_layers).to(device)
        else:
            model = cLSTM(p, cfg.lag, cfg.hidden).to(device)

        self._model = model

        # ---- prepare data -----------------------------------------------
        X_seq, Y_seq = self._build_sequences(data, cfg.lag)   # (N, lag, p), (N, p)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_seq, dtype=torch.float32)

        N = X_tensor.shape[0]
        n_val = max(1, int(N * cfg.val_fraction))
        n_train = N - n_val

        X_train, Y_train = X_tensor[:n_train].to(device), Y_tensor[:n_train].to(device)
        X_val,   Y_val   = X_tensor[n_train:].to(device), Y_tensor[n_train:].to(device)

        # ---- optimiser --------------------------------------------------
        opt = Adam(model.parameters(), lr=cfg.lr)
        scheduler = ReduceLROnPlateau(opt, patience=10, factor=0.5)

        loss_history: Dict[str, List[float]] = {"total": [], "mse": [], "reg": []}
        weights_history: List[np.ndarray] = []

        best_val = float("inf")
        patience_counter = 0
        n_epochs_trained = 0

        # ---- training loop ----------------------------------------------
        for epoch in range(1, cfg.max_epochs + 1):
            model.train()
            idx = torch.randperm(n_train, device=device)
            epoch_mse = epoch_reg = 0.0
            n_batches = 0

            for start in range(0, n_train, cfg.batch_size):
                b_idx = idx[start: start + cfg.batch_size]
                xb, yb = X_train[b_idx], Y_train[b_idx]

                pred = model(xb)
                mse  = nn.functional.mse_loss(pred, yb)
                reg  = (
                    cfg.lambda_group * model.group_lasso_penalty()
                    + cfg.lambda_ridge * self._ridge_penalty(model)
                )
                loss = mse + reg

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()

                epoch_mse += mse.item()
                epoch_reg += reg.item()
                n_batches += 1

            avg_mse = epoch_mse / max(n_batches, 1)
            avg_reg = epoch_reg / max(n_batches, 1)
            avg_tot = avg_mse + avg_reg

            # validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = nn.functional.mse_loss(val_pred, Y_val).item()

            scheduler.step(val_loss)

            loss_history["total"].append(avg_tot)
            loss_history["mse"].append(avg_mse)
            loss_history["reg"].append(avg_reg)

            if epoch % cfg.record_every == 0:
                weights_history.append(model.get_adjacency())

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            n_epochs_trained = epoch

            if cfg.verbose and epoch % 50 == 0:
                logger.info(
                    f"Epoch {epoch:4d} | train_loss={avg_tot:.5f} "
                    f"(mse={avg_mse:.5f}, reg={avg_reg:.5f}) | val_mse={val_loss:.5f}"
                )

            if patience_counter >= cfg.patience:
                if cfg.verbose:
                    logger.info(f"Early stopping at epoch {epoch}.")
                break

        # ---- extract causal structure -----------------------------------
        model.eval()
        adjacency = model.get_adjacency()   # (p, p) — đã fill_diagonal(0) bên trong
        causal = (adjacency > cfg.threshold).astype(bool)
        np.fill_diagonal(causal, False)     # đảm bảo không có self-loop

        self._result = NeuralGrangerResult(
            adjacency_matrix=adjacency,
            causal_matrix=causal,
            weights_history=weights_history,
            loss_history=loss_history,
            feature_names=feature_names,
            model_type=cfg.model_type,
            threshold=cfg.threshold,
            n_epochs_trained=n_epochs_trained,
        )
        return self._result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sequences(
        data: np.ndarray, lag: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts (T, p) time series into supervised sequences.
        X : (T - lag, lag, p)
        Y : (T - lag, p)   — next-step targets
        """
        T, p = data.shape
        X_list, Y_list = [], []
        for t in range(lag, T):
            X_list.append(data[t - lag: t])   # (lag, p)
            Y_list.append(data[t])             # (p,)
        return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)

    @staticmethod
    def _ridge_penalty(model: nn.Module) -> torch.Tensor:
        """L2 penalty on ALL parameters (including biases)."""
        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        for param in model.parameters():
            penalty = penalty + param.pow(2).sum()
        return penalty

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_risk_scores(self) -> Optional[Dict[str, float]]:
        """
        Compute a simple Risk Score per variable = (out-degree strength).
        Returns None if fit() has not been called.
        """
        if self._result is None:
            return None
        A = self._result.adjacency_matrix
        names = self._result.feature_names
        # out-degree = sum of row i  (i causes others)
        scores = {name: float(A[i].sum()) for i, name in enumerate(names)}
        return scores

    def get_causal_edges(self) -> List[Tuple[str, str, float]]:
        """
        Returns list of (cause, effect, strength) for all detected edges.
        """
        if self._result is None:
            return []
        A = self._result.adjacency_matrix
        C = self._result.causal_matrix
        names = self._result.feature_names
        edges = []
        for i in range(len(names)):
            for j in range(len(names)):
                if C[i, j] and i != j:
                    # A[i,j] means j -> i
                    edges.append((names[j], names[i], float(A[i, j])))
        edges.sort(key=lambda e: -e[2])
        return edges

    @property
    def result(self) -> Optional[NeuralGrangerResult]:
        return self._result


# ---------------------------------------------------------------------------
# Convenience wrapper: run both cMLP and cLSTM and compare
# ---------------------------------------------------------------------------

def compare_architectures(
    data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    base_config: Optional[NeuralGrangerConfig] = None,
) -> Dict[str, NeuralGrangerResult]:
    """
    Train cMLP and cLSTM on the same data and return both results.

    Parameters
    ----------
    data : np.ndarray  shape (T, p)
    feature_names : optional list of variable names
    base_config : base NeuralGrangerConfig (model_type is overridden)

    Returns
    -------
    dict with keys 'cMLP' and 'cLSTM'
    """
    base_config = base_config or NeuralGrangerConfig()
    results: Dict[str, NeuralGrangerResult] = {}

    for arch in ("cMLP", "cLSTM"):
        cfg = NeuralGrangerConfig(
            model_type   = arch,          # type: ignore[arg-type]
            lag          = base_config.lag,
            hidden       = base_config.hidden,
            n_layers     = base_config.n_layers,
            lambda_group = base_config.lambda_group,
            lambda_ridge = base_config.lambda_ridge,
            lr           = base_config.lr,
            max_epochs   = base_config.max_epochs,
            patience     = base_config.patience,
            batch_size   = base_config.batch_size,
            val_fraction = base_config.val_fraction,
            threshold    = base_config.threshold,
            device       = base_config.device,
            seed         = base_config.seed,
            record_every = base_config.record_every,
            verbose      = base_config.verbose,
        )
        logger.info(f"--- Training {arch} ---")
        estimator = NeuralGrangerCausality(cfg)
        results[arch] = estimator.fit(data, feature_names=feature_names)
        logger.info(results[arch].summary())

    return results