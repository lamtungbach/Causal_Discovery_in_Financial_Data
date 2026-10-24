from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import RETURN_COLS, LABELS, TABLE_DIR, FIGURE_DIR
from src.common import save_table


def read_adj(path):
    df = pd.read_csv(path)
    labels = df.iloc[:, 0].tolist()
    mat = df.iloc[:, 1:].values.astype(float)
    return labels, mat


def main():
    adj_path = TABLE_DIR / "08_neural_granger_adjacency_thresholded.csv"
    if not adj_path.exists():
        raise FileNotFoundError("Chưa có adjacency. Chạy scripts/08_neural_granger.py trước.")
    labels, adj = read_adj(adj_path)
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("Cần cài networkx: pip install networkx") from e
    G = nx.DiGraph()
    for lab in labels: G.add_node(lab)
    for i, src in enumerate(labels):
        for j, tgt in enumerate(labels):
            if i != j and adj[i, j] > 0:
                G.add_edge(src, tgt, weight=float(adj[i, j]))
    rows = []
    pagerank = nx.pagerank(G, weight="weight") if len(G) else {}
    bet = nx.betweenness_centrality(G, weight="weight") if len(G) else {}
    for node in labels:
        out_w = sum(d.get("weight", 1.0) for _, _, d in G.out_edges(node, data=True))
        in_w = sum(d.get("weight", 1.0) for _, _, d in G.in_edges(node, data=True))
        rows.append({
            "node": node,
            "in_degree": G.in_degree(node), "out_degree": G.out_degree(node),
            "weighted_in": in_w, "weighted_out": out_w,
            "net_transmitter_score": out_w - in_w,
            "pagerank": pagerank.get(node, 0), "betweenness": bet.get(node, 0),
        })
    save_table(pd.DataFrame(rows).sort_values("net_transmitter_score", ascending=False), TABLE_DIR / "09_graph_node_metrics.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42, weight="weight")
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=1200, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
    nx.draw_networkx_edges(G, pos, width=[1 + 3*w/(max(weights) if weights else 1) for w in weights], arrows=True, arrowstyle="-|>", ax=ax)
    ax.set_title("Neural Granger spillover graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "09_neural_granger_graph.png", dpi=170)
    plt.close(fig)
    print("✓ Graph analytics completed")

if __name__ == "__main__":
    main()
