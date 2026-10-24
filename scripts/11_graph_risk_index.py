from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import RETURN_FILE, RETURN_COLS, LABELS, TARGET_COL, TABLE_DIR, FIGURE_DIR
from src.common import load_timeseries, save_table


def percentile_rank(x):
    s = pd.Series(x)
    return s.rank(pct=True).values * 100


def main(window=63):
    df = load_timeseries(RETURN_FILE).dropna().reset_index(drop=True)
    # Rolling graph-risk proxy: mean absolute lead-lag correlation into VNI + VNI volatility.
    target = TARGET_COL
    records = []
    for t in range(window, len(df)-1):
        sub = df.iloc[t-window:t+1]
        incoming = 0.0
        for src in RETURN_COLS:
            if src == target: continue
            x = sub[src].iloc[:-1].values; y = sub[target].iloc[1:].values
            if np.std(x) > 0 and np.std(y) > 0:
                incoming += abs(np.corrcoef(x, y)[0,1])
        vni_vol = sub[target].std()
        records.append({"date": df["date"].iloc[t], "incoming_vni_leadlag": incoming, "vni_volatility": vni_vol})
    out = pd.DataFrame(records)
    out["incoming_score_pct"] = percentile_rank(out["incoming_vni_leadlag"])
    out["vol_score_pct"] = percentile_rank(out["vni_volatility"])
    out["graph_based_risk_index"] = 0.6 * out["incoming_score_pct"] + 0.4 * out["vol_score_pct"]
    out["risk_regime"] = pd.cut(out["graph_based_risk_index"], bins=[-1, 60, 80, 95, 101], labels=["Low", "Medium", "High", "Extreme"])
    save_table(out, TABLE_DIR / "11_graph_based_risk_index.csv")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(out["date"], out["graph_based_risk_index"])
    ax.axhline(80, linestyle="--", linewidth=1)
    ax.axhline(95, linestyle="--", linewidth=1)
    ax.set_title("Graph-based Risk Index for VN-Index")
    ax.set_ylabel("Risk score percentile")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "11_graph_based_risk_index.png", dpi=160)
    plt.close(fig)
    print("✓ Graph-based risk index completed")

if __name__ == "__main__":
    main()
