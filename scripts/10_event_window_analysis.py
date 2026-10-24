from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import RETURN_FILE, RETURN_COLS, LABELS, TABLE_DIR, EVENT_WINDOWS
from src.common import load_timeseries, save_table, split_by_period


def simple_window_graph(sub, threshold=0.25):
    # Graph proxy dùng lead-lag correlation 1 ngày: corr(src_t, target_{t+1})
    mat = np.zeros((len(RETURN_COLS), len(RETURN_COLS)))
    for i, src in enumerate(RETURN_COLS):
        for j, tgt in enumerate(RETURN_COLS):
            if i == j: continue
            x = sub[src].iloc[:-1].values
            y = sub[tgt].iloc[1:].values
            if len(x) > 5 and np.std(x) > 0 and np.std(y) > 0:
                c = np.corrcoef(x, y)[0, 1]
                mat[i, j] = abs(c) if abs(c) >= threshold else 0.0
    return mat


def main():
    df = load_timeseries(RETURN_FILE).dropna().reset_index(drop=True)
    labels = [LABELS[c] for c in RETURN_COLS]
    rows = []
    for name, (start, end) in EVENT_WINDOWS.items():
        sub = split_by_period(df, start, end)
        if len(sub) < 30:
            continue
        mat = simple_window_graph(sub)
        density = np.count_nonzero(mat) / (mat.shape[0] * (mat.shape[0]-1))
        vni_idx = labels.index("VNI")
        incoming_vni = float(mat[:, vni_idx].sum())
        outgoing_vni = float(mat[vni_idx, :].sum())
        rows.append({"event": name, "start": start, "end": end, "n": len(sub), "graph_density": density, "incoming_to_vni": incoming_vni, "outgoing_from_vni": outgoing_vni})
        save_table(pd.DataFrame(mat, index=labels, columns=labels).reset_index().rename(columns={"index":"source"}), TABLE_DIR / f"10_event_adj_{name}.csv")
    save_table(pd.DataFrame(rows), TABLE_DIR / "10_event_window_summary.csv")
    print("✓ Event window analysis completed")

if __name__ == "__main__":
    main()
