from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import RETURN_FILE, RETURN_COLS, LABELS, TABLE_DIR, FIGURE_DIR
from src.common import load_timeseries, save_table


def jb_pvalue(x):
    try:
        from scipy.stats import jarque_bera
        return jarque_bera(x).statistic, jarque_bera(x).pvalue
    except Exception:
        return np.nan, np.nan


def adf_test(x):
    try:
        from statsmodels.tsa.stattools import adfuller
        res = adfuller(x, autolag="AIC")
        return res[0], res[1]
    except Exception:
        return np.nan, np.nan


def main():
    df = load_timeseries(RETURN_FILE).dropna()
    desc = []
    for c in RETURN_COLS:
        x = df[c].dropna().values
        jb, jbp = jb_pvalue(x)
        adf, adfp = adf_test(x)
        desc.append({
            "Variable": LABELS.get(c, c), "N": len(x), "Mean": np.mean(x), "Std": np.std(x, ddof=1),
            "Min": np.min(x), "Max": np.max(x), "Skewness": pd.Series(x).skew(),
            "Kurtosis": pd.Series(x).kurt(), "JB Stat": jb, "JB p-value": jbp,
            "ADF Stat": adf, "ADF p-value": adfp,
        })
    save_table(pd.DataFrame(desc), TABLE_DIR / "03_descriptive_stationarity.csv")
    corr = df[RETURN_COLS].corr()
    corr.index = [LABELS.get(c, c) for c in RETURN_COLS]
    corr.columns = [LABELS.get(c, c) for c in RETURN_COLS]
    save_table(corr.reset_index().rename(columns={"index":"Variable"}), TABLE_DIR / "03_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation matrix of returns")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_correlation_heatmap.png", dpi=160)
    plt.close(fig)
    print("✓ EDA completed")

if __name__ == "__main__":
    main()
