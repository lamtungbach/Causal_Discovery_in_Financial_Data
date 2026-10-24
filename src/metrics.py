import numpy as np


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else np.nan
    direction_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Directional_Accuracy": direction_acc}


def adjacency_metrics(adj):
    adj = np.asarray(adj)
    off_diag = adj.copy()
    np.fill_diagonal(off_diag, 0)
    density = float(np.count_nonzero(off_diag) / (off_diag.shape[0] * (off_diag.shape[0] - 1)))
    sparsity = 1.0 - density
    return {"density": density, "sparsity": sparsity, "num_edges": int(np.count_nonzero(off_diag))}


def jaccard_edges(a, b, threshold=1e-8):
    a = np.asarray(a).copy(); b = np.asarray(b).copy()
    np.fill_diagonal(a, 0); np.fill_diagonal(b, 0)
    ea = set(map(tuple, np.argwhere(np.abs(a) > threshold)))
    eb = set(map(tuple, np.argwhere(np.abs(b) > threshold)))
    if not ea and not eb:
        return 1.0
    return len(ea & eb) / max(1, len(ea | eb))
