"""
src/models/notears.py
Chức năng: Chạy NOTEARS — thuật toán chính của khóa luận.
Hỗ trợ notears v3.0 (linear) và fallback tự implement.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import (NOTEARS_INPUT, NOTEARS_LAMBDA, NOTEARS_THRESH,
                    LABELS, PATH_RESULTS)


def load_data() -> tuple:
    """Đọc và chuẩn hóa dữ liệu cho NOTEARS."""
    df   = pd.read_csv(NOTEARS_INPUT, index_col=0, parse_dates=True)
    df   = df.dropna()
    cols = df.columns.tolist()
    print(f"   Dữ liệu: {df.shape[0]} ngày × {df.shape[1]} thị trường")

    # Chuẩn hóa về mean=0, std=1
    X = df.values.astype(float)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return df, X, cols


def _get_notears_func():
    """
    Thử import notears theo nhiều cách khác nhau.
    notears v3.0 đổi tên hàm so với v1.
    """
    # Cách 1: notears v1 (notears_linear)
    try:
        from notears import notears_linear
        print("   Dùng: notears v1 (notears_linear)")
        return notears_linear
    except ImportError:
        pass

    # Cách 2: notears v3.0 (linear)
    try:
        from notears.linear import notears_linear
        print("   Dùng: notears v3.0 (notears.linear)")
        return notears_linear
    except ImportError:
        pass

    # Cách 3: notears v3.0 (lộ tên khác)
    try:
        import notears
        funcs = [f for f in dir(notears) if "linear" in f.lower() or "notears" in f.lower()]
        if funcs:
            fn = getattr(notears, funcs[0])
            print(f"   Dùng: notears.{funcs[0]}")
            return fn
    except Exception:
        pass

    print("   ⚠️  Không tìm thấy hàm notears — dùng bản tự implement")
    return None


def _notears_simple(X: np.ndarray, lambda1: float) -> np.ndarray:
    """
    Bản implement NOTEARS đơn giản bằng scipy (fallback).
    Tham khảo: Zheng et al. (2018)
    """
    from scipy.optimize import minimize
    n, d = X.shape

    def _h(W):
        M = np.eye(d) + W * W / d
        E = np.linalg.matrix_power(M, d)
        return np.trace(E) - d

    rho, alpha, W_est = 1.0, 0.0, np.zeros((d, d))

    for iteration in range(100):
        def _func(w_flat):
            W    = w_flat.reshape(d, d)
            R    = X - X @ W
            loss = 0.5 / n * (R ** 2).sum()
            h    = _h(W)
            reg  = lambda1 * np.abs(W).sum()
            return loss + reg + 0.5 * rho * h ** 2 + alpha * h

        sol   = minimize(_func, W_est.flatten(), method="L-BFGS-B",
                         options={"maxiter": 200, "ftol": 1e-12})
        W_new = sol.x.reshape(d, d)
        h_val = _h(W_new)

        if abs(h_val) < 1e-8:
            W_est = W_new
            break
        alpha += rho * h_val
        if abs(h_val) > 0.25 * abs(_h(W_est)):
            rho = min(rho * 10, 1e16)
        W_est = W_new

    print(f"   Hoàn thành sau {iteration+1} vòng lặp, h={_h(W_est):.2e}")
    return W_est


def run_notears():
    """Entry point — chạy NOTEARS và lưu kết quả."""
    print("\n⭐ [NOTEARS — Thuật toán chính]")
    os.makedirs(PATH_RESULTS, exist_ok=True)

    df, X, cols = load_data()

    # Lấy hàm notears phù hợp
    notears_fn = _get_notears_func()

    print(f"   Đang tối ưu (lambda={NOTEARS_LAMBDA})...")
    if notears_fn is not None:
        try:
            W_est = notears_fn(X, lambda1=NOTEARS_LAMBDA, loss_type="l2")
        except TypeError:
            # Một số version dùng tên tham số khác
            try:
                W_est = notears_fn(X, NOTEARS_LAMBDA)
            except Exception as e:
                print(f"   ⚠️  Lỗi khi chạy notears: {e}")
                print("   Chuyển sang dùng bản tự implement...")
                W_est = _notears_simple(X, NOTEARS_LAMBDA)
    else:
        W_est = _notears_simple(X, NOTEARS_LAMBDA)

    # Áp dụng ngưỡng cắt cạnh
    W_thresh = W_est.copy()
    W_thresh[np.abs(W_thresh) < NOTEARS_THRESH] = 0

    # DAG nhị phân
    dag_bin = (np.abs(W_thresh) > 0).astype(int)
    np.fill_diagonal(dag_bin, 0)

    # Tạo DataFrame có tên đầy đủ
    W_df      = pd.DataFrame(W_thresh, index=cols, columns=cols)
    dag_df    = pd.DataFrame(dag_bin,  index=cols, columns=cols)
    W_named   = W_df.rename(index=LABELS, columns=LABELS)
    dag_named = dag_df.rename(index=LABELS, columns=LABELS)

    # Lưu kết quả
    W_named.to_csv(os.path.join(PATH_RESULTS,   "notears_weights.csv"))
    dag_named.to_csv(os.path.join(PATH_RESULTS, "notears_dag.csv"))

    # In tóm tắt cạnh
    edges = []
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if dag_bin[i, j] == 1:
                edges.append((LABELS.get(ci, ci),
                               LABELS.get(cj, cj),
                               round(W_thresh[i, j], 4)))

    print(f"\n   Số cạnh (threshold={NOTEARS_THRESH}): {len(edges)}")
    if edges:
        print("   Các cạnh (sắp xếp theo cường độ):")
        for cause, caused, w in sorted(edges, key=lambda x: abs(x[2]), reverse=True):
            bar = "█" * min(int(abs(w) * 15), 20)
            print(f"     {cause:<20} →  {caused:<20}  w={w:+.4f}  {bar}")

    # Phân tích in/out degree
    print("\n   Vai trò các thị trường:")
    for i, col in enumerate(cols):
        out  = int(dag_bin[i, :].sum())
        inn  = int(dag_bin[:, i].sum())
        role = ("NGUỒN RỦI RO" if out > inn
                else "NHẬN RỦI RO" if inn > out
                else "Trung gian")
        print(f"     {LABELS.get(col,col):<20}  out={out}  in={inn}  → {role}")

    print("\n   ✅ Đã lưu:")
    print(f"      {PATH_RESULTS}/notears_weights.csv")
    print(f"      {PATH_RESULTS}/notears_dag.csv")
    return W_df, dag_df


if __name__ == "__main__":
    run_notears()
