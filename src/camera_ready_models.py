from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Required by CUDA/cuBLAS when deterministic algorithms are enabled. This must
# be set before the first CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str | None) -> str:
    if requested in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return requested


class TargetMLP(nn.Module):
    """Target-specific nonlinear autoregression.

    Inputs remain [sample, lag, source]. Flattening is performed only inside
    forward(), so source ablation can operate on an unambiguous tensor axis.
    """

    def __init__(self, n_sources: int, n_lags: int, hidden: tuple[int, int]):
        super().__init__()
        h1, h2 = hidden
        self.n_sources = n_sources
        self.n_lags = n_lags
        self.network = nn.Sequential(
            nn.Linear(n_sources * n_lags, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.reshape(len(x), -1)).squeeze(-1)

    @property
    def input_weight(self) -> torch.Tensor:
        return self.network[0].weight


class VniLSTM(nn.Module):
    def __init__(self, n_sources: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(n_sources, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.output(sequence[:, -1]).squeeze(-1)


@dataclass
class FitResult:
    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_val_mse: float


def _fit_single_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: str,
    epochs: int,
    patience: int,
    lr: float,
    penalty_fn=None,
    min_delta: float = 1e-7,
) -> FitResult:
    model = model.to(device)
    Xtr = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    Xva = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yva = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        prediction = model(Xtr)
        data_loss = mse(prediction, ytr)
        penalty = penalty_fn(model) if penalty_fn is not None else torch.zeros((), device=device)
        objective = data_loss + penalty
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mse = float(mse(model(Xva), yva).item())
        history.append(
            {
                "epoch": epoch,
                "train_mse": float(data_loss.item()),
                "regularized_objective": float(objective.item()),
                "val_mse": val_mse,
            }
        )
        if val_mse < best_val - min_delta:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    return FitResult(
        model=model,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        best_val_mse=best_val,
    )


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden: int,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
    lr: float,
) -> FitResult:
    configure_reproducibility(seed)
    model = VniLSTM(X_train.shape[-1], hidden)
    return _fit_single_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        device=device,
        epochs=epochs,
        patience=patience,
        lr=lr,
    )


def _group_norm_penalty(model: TargetMLP, lam: float) -> torch.Tensor:
    """Differentiable group-norm regularizer for lag-major flattened inputs.

    This is regularization under Adam, not a proximal exact-zero selection
    operator. Sparse graph selection is performed later using validation
    prediction ablation.
    """
    weight = model.input_weight
    penalty = torch.zeros((), dtype=weight.dtype, device=weight.device)
    for source in range(model.n_sources):
        indices = torch.arange(
            source,
            model.n_sources * model.n_lags,
            model.n_sources,
            device=weight.device,
        )
        group = weight.index_select(1, indices)
        penalty = penalty + torch.linalg.vector_norm(group)
    return lam * penalty


def train_ngc(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    *,
    hidden: tuple[int, int],
    lam: float,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
    lr: float,
) -> tuple[list[TargetMLP], pd.DataFrame, pd.DataFrame]:
    configure_reproducibility(seed)
    models: list[TargetMLP] = []
    histories = []
    checkpoints = []
    for target in range(Y_train.shape[1]):
        # A target-specific offset keeps initialization deterministic while
        # avoiding identical initial weights for every target network.
        configure_reproducibility(seed * 1009 + target)
        model = TargetMLP(X_train.shape[-1], X_train.shape[1], hidden)
        result = _fit_single_model(
            model,
            X_train,
            Y_train[:, target],
            X_val,
            Y_val[:, target],
            device=device,
            epochs=epochs,
            patience=patience,
            lr=lr,
            penalty_fn=lambda fitted, lam=lam: _group_norm_penalty(fitted, lam),
        )
        history = result.history.copy()
        history["target_index"] = target
        histories.append(history)
        checkpoints.append(
            {
                "target_index": target,
                "best_epoch": result.best_epoch,
                "best_val_mse": result.best_val_mse,
                "epochs_run": len(result.history),
            }
        )
        models.append(result.model)
        if device == "cuda":
            torch.cuda.empty_cache()
    return models, pd.concat(histories, ignore_index=True), pd.DataFrame(checkpoints)


def predict_model(model: nn.Module, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model = model.cpu().eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32)
            predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=float)


def predict_ngc(models: list[TargetMLP], X: np.ndarray) -> np.ndarray:
    return np.column_stack([predict_model(model, X) for model in models])


def block_ablation_importance(
    models: list[TargetMLP],
    X_val: np.ndarray,
    Y_val: np.ndarray,
    labels: list[str],
    eps: float = 1e-12,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Validation prediction importance from ablating one complete source.

    The unflattened source axis is used directly; no contiguous-block layout
    assumption is made. Signed loss changes are preserved in the audit table.
    """
    n_sources = X_val.shape[-1]
    signed_matrix = np.zeros((n_sources, n_sources), dtype=float)
    rows = []
    for target, model in enumerate(models):
        full_prediction = predict_model(model, X_val)
        full_mse = float(np.mean((Y_val[:, target] - full_prediction) ** 2))
        if not np.isfinite(full_mse):
            raise ValueError(f"Non-finite full validation loss for target {labels[target]}")
        for source in range(n_sources):
            if source == target:
                continue
            ablated = X_val.copy()
            ablated[:, :, source] = 0.0
            ablated_prediction = predict_model(model, ablated)
            ablated_mse = float(np.mean((Y_val[:, target] - ablated_prediction) ** 2))
            signed_change = (ablated_mse - full_mse) / (full_mse + eps)
            positive_importance = max(0.0, signed_change)
            signed_matrix[source, target] = positive_importance
            rows.append(
                {
                    "source": labels[source],
                    "target": labels[target],
                    "full_val_mse": full_mse,
                    "ablated_val_mse": ablated_mse,
                    "signed_relative_loss_change": signed_change,
                    "relative_loss_increase": positive_importance,
                }
            )
    return signed_matrix, pd.DataFrame(rows)


def sparsify_importance(
    importance: np.ndarray, quantile: float
) -> tuple[np.ndarray, float]:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must lie in [0, 1]")
    matrix = np.asarray(importance, dtype=float).copy()
    np.fill_diagonal(matrix, 0.0)
    positive = matrix[matrix > 0]
    if len(positive) == 0:
        return np.zeros_like(matrix), float("nan")
    threshold = float(np.quantile(positive, quantile))
    retained = np.where(matrix >= threshold, matrix, 0.0)
    np.fill_diagonal(retained, 0.0)
    return retained, threshold
