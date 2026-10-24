from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import RETURN_FILE, RETURN_COLS, LABELS, TABLE_DIR
from src.common import load_timeseries, save_table


def main(max_lag=5, alpha=0.05):
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError as e:
        raise ImportError("Cần cài statsmodels: pip install statsmodels") from e
    df = load_timeseries(RETURN_FILE).dropna().reset_index(drop=True)
    n = len(RETURN_COLS)
    adj = np.zeros((n, n))
    pvals = np.ones((n, n))
    rows = []
    for i, src in enumerate(RETURN_COLS):
        for j, tgt in enumerate(RETURN_COLS):
            if i == j:
                continue
            # grangercausalitytests expects columns [target, source]
            data = df[[tgt, src]].dropna()
            try:
                res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                lag_pvals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]
                p = float(np.min(lag_pvals))
            except Exception:
                p = 1.0
            pvals[i, j] = p
            adj[i, j] = 1.0 if p < alpha else 0.0
            rows.append({"source": LABELS.get(src, src), "target": LABELS.get(tgt, tgt), "min_pvalue": p, "edge": p < alpha})
    save_table(pd.DataFrame(rows), TABLE_DIR / "06_var_granger_edges.csv")
    mat = pd.DataFrame(adj, index=[LABELS[c] for c in RETURN_COLS], columns=[LABELS[c] for c in RETURN_COLS])
    save_table(mat.reset_index().rename(columns={"index":"source"}), TABLE_DIR / "06_var_granger_adjacency.csv")
    pmat = pd.DataFrame(pvals, index=[LABELS[c] for c in RETURN_COLS], columns=[LABELS[c] for c in RETURN_COLS])
    save_table(pmat.reset_index().rename(columns={"index":"source"}), TABLE_DIR / "06_var_granger_pvalues.csv")
    print("✓ VAR/linear Granger baseline completed")

if __name__ == "__main__":
    main()
