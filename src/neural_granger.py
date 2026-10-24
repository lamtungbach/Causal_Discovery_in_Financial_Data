"""Archived lightweight neural-Granger implementation.

The KSE camera-ready pipeline uses :mod:`src.camera_ready_models` instead.
"""

from __future__ import annotations
import numpy as np


def _torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as e:
        raise ImportError("Cần cài PyTorch: pip install torch") from e


class LaggedMLPWrapper:
    """Neural Granger đơn giản: MLP dự báo từng target từ lagged multivariate inputs.

    Group lasso trên trọng số input theo từng biến nguồn giúp chọn cạnh Granger phi tuyến.
    Đây là phiên bản nhẹ, dễ chạy cho khóa luận tốt nghiệp AI.
    """
    def __init__(self, n_vars, lags=5, hidden=32, lr=1e-3, lam=1e-3, epochs=80, patience=8, seed=42, device=None, verbose=True):
        self.n_vars = n_vars
        self.lags = lags
        self.hidden = hidden
        self.lr = lr
        self.lam = lam
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.device = device
        self.models = []
        self.histories = []
        self.verbose = verbose

    def _build(self):
        torch, nn = _torch()
        return nn.Sequential(
            nn.Linear(self.n_vars * self.lags, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def _group_lasso(self, model):
        torch, _ = _torch()
        W = model[0].weight  # [hidden, n_vars*lags]
        penalty = 0.0
        for src in range(self.n_vars):
            idx = []
            for lag in range(self.lags):
                idx.append(lag * self.n_vars + src)
            group = W[:, idx]
            penalty = penalty + torch.sqrt(torch.sum(group ** 2) + 1e-8)
        return penalty

    def fit(self, X_train, Y_train, X_val=None, Y_val=None):
        torch, nn = _torch()
        torch.manual_seed(self.seed)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        X_train_t = torch.tensor(X_train.reshape(len(X_train), -1), dtype=torch.float32, device=device)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val.reshape(len(X_val), -1), dtype=torch.float32, device=device) if X_val is not None else None
        Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=device) if Y_val is not None else None

        self.models = []
        self.histories = []
        for target in range(self.n_vars):
            model = self._build().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()
            best_state = None
            best_val = float("inf")
            bad = 0
            hist = []
            for ep in range(self.epochs):
                model.train()
                pred = model(X_train_t).squeeze(-1)
                loss = loss_fn(pred, Y_train_t[:, target]) + self.lam * self._group_lasso(model)
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    if X_val_t is not None:
                        val = loss_fn(model(X_val_t).squeeze(-1), Y_val_t[:, target]).item()
                    else:
                        val = loss.item()
                hist.append({"epoch": ep + 1, "target": target, "train_loss": float(loss.item()), "val_loss": float(val)})
                if val < best_val - 1e-7:
                    best_val = val
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                if bad >= self.patience:
                    break
            if best_state is not None:
                model.load_state_dict(best_state)
            self.models.append(model.cpu())
            self.histories.extend(hist)
        return self

    def adjacency(self):
        """adj[src, target] = norm trọng số source trong model dự báo target."""
        adj = np.zeros((self.n_vars, self.n_vars), dtype=float)
        for target, model in enumerate(self.models):
            W = model[0].weight.detach().cpu().numpy()
            for src in range(self.n_vars):
                idx = [lag * self.n_vars + src for lag in range(self.lags)]
                adj[src, target] = float(np.sqrt(np.sum(W[:, idx] ** 2)))
        np.fill_diagonal(adj, 0.0)
        return adj

    def predict_all(self, X):
        torch, _ = _torch()
        X_t = torch.tensor(X.reshape(len(X), -1), dtype=torch.float32)
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                preds.append(model(X_t).squeeze(-1).numpy())
        return np.stack(preds, axis=1)
