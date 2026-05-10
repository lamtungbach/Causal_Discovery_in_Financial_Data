"""
src/visualization/causal_graph.py
Vẽ DAG nhân quả cho từng phương pháp.

Đặt file tại: src/visualization/causal_graph.py
"""

import numpy as np
import logging
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Color & Label mặc định (override bằng config nếu có) ────────────────────
_COLORS = {
    "SP500"  : "#E63946",
    "VNINDEX": "#F4A261",
    "Gold"   : "#FFD166",
    "WTI_Oil": "#06D6A0",
    "Bitcoin": "#118AB2",
}
_LABELS = {
    "SP500"  : "S&P 500",
    "VNINDEX": "VN-Index",
    "Gold"   : "Vàng (Gold)",
    "WTI_Oil": "Dầu WTI",
    "Bitcoin": "Bitcoin",
}

try:
    from config import MARKET_COLORS as _COLORS, MARKET_LABELS as _LABELS, FIG_DPI
except ImportError:
    FIG_DPI = 150


# ─────────────────────────────────────────────────────────────────────────────

def plot_dag(
    binary_matrix : np.ndarray,
    node_names    : List[str],
    weight_matrix : Optional[np.ndarray] = None,
    title         : str  = "Causal DAG",
    output_path   : Optional[str] = None,
    figsize       : tuple = (9, 7),
    show_weights  : bool  = True,
    layout        : str   = "circular",
) -> "plt.Figure":
    """
    Vẽ DAG nhân quả với đầy đủ thông tin trực quan.

    Visual encoding
    ---------------
    - Màu node          → thị trường (theo MARKET_COLORS)
    - Kích thước node   → out-degree (hub node to hơn)
    - Độ dày cạnh       → |trọng số| (nếu có weight_matrix)
    - Nhãn trên cạnh    → giá trị trọng số (nếu show_weights=True)
    - Mũi tên           → hướng nhân quả (source → target)

    Parameters
    ----------
    binary_matrix : np.ndarray (d, d)
        Ma trận 0/1 — chỉ vẽ các cạnh = 1.
    node_names : List[str]
        Tên các thị trường.
    weight_matrix : np.ndarray (d, d), optional
        Ma trận trọng số W từ NOTEARS.
    title : str
        Tiêu đề đồ thị.
    output_path : str, optional
        Đường dẫn lưu file ảnh. Nếu None thì không lưu.
    figsize : tuple
        Kích thước figure.
    show_weights : bool
        Có hiển thị nhãn trọng số trên cạnh không.
    layout : str
        "circular" | "spring" | "shell" | "kamada_kawai"

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> fig = plot_dag(
    ...     binary_matrix = notears_result.binary_matrix,
    ...     node_names    = ["SP500","VNINDEX","Gold","WTI_Oil","Bitcoin"],
    ...     weight_matrix = notears_result.adjacency_matrix,
    ...     title         = "NOTEARS — Causal DAG (threshold=0.2)",
    ...     output_path   = "reports/figures/dag_notears.png",
    ... )
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx
    except ImportError:
        raise ImportError("pip install matplotlib networkx")

    # ── Xây dựng đồ thị ───────────────────────────────────────────────────────
    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    for i, src in enumerate(node_names):
        for j, tgt in enumerate(node_names):
            if binary_matrix[i, j] == 1:
                w = abs(float(weight_matrix[i, j])) if weight_matrix is not None else 1.0
                G.add_edge(src, tgt, weight=w)

    # ── Layout ────────────────────────────────────────────────────────────────
    _layout_map = {
        "circular"    : nx.circular_layout,
        "spring"      : nx.spring_layout,
        "shell"       : nx.shell_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
    }
    pos = _layout_map.get(layout, nx.circular_layout)(G)

    # ── Visual attributes ─────────────────────────────────────────────────────
    node_colors = [_COLORS.get(n, "#aaaaaa") for n in G.nodes()]
    out_deg     = dict(G.out_degree())
    node_sizes  = [900 + out_deg.get(n, 0) * 380 for n in G.nodes()]

    edge_ws     = [G[u][v]["weight"] for u, v in G.edges()] if G.number_of_edges() > 0 else [1]
    max_w       = max(edge_ws) if max(edge_ws) > 0 else 1.0
    edge_widths = [1.5 + 4.0 * (G[u][v]["weight"] / max_w) for u, v in G.edges()]

    # ── Vẽ ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(
        G, pos,
        node_color = node_colors,
        node_size  = node_sizes,
        alpha      = 0.92,
        ax         = ax,
    )
    nx.draw_networkx_labels(
        G, pos,
        labels      = {n: _LABELS.get(n, n) for n in G.nodes()},
        font_size   = 10,
        font_weight = "bold",
        ax          = ax,
    )
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(
            G, pos,
            width           = edge_widths,
            edge_color      = "#333333",
            arrows          = True,
            arrowsize       = 22,
            connectionstyle = "arc3,rad=0.08",
            ax              = ax,
            min_source_margin = 20,
            min_target_margin = 20,
        )

        if show_weights and weight_matrix is not None:
            edge_labels = {
                (u, v): f"{G[u][v]['weight']:.2f}"
                for u, v in G.edges()
            }
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels = edge_labels,
                font_size   = 8,
                bbox        = dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6),
                ax          = ax,
            )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=_COLORS.get(n, "#aaa"), label=_LABELS.get(n, n))
        for n in node_names
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9, framealpha=0.8)

    # ── Info text ─────────────────────────────────────────────────────────────
    n_edges = int(binary_matrix.sum())
    ax.text(
        0.99, 0.01,
        f"Edges: {n_edges}  |  Max out-degree: {int(max(out_deg.values(), default=0))}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="gray",
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.axis("off")
    plt.tight_layout()

    _save_fig(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────

def plot_adjacency_heatmap(
    weight_matrix: np.ndarray,
    node_names   : List[str],
    title        : str = "Adjacency Matrix Heatmap — NOTEARS W",
    output_path  : Optional[str] = None,
) -> "plt.Figure":
    """
    Heatmap ma trận trọng số W từ NOTEARS (trước threshold).
    Trực quan hóa cường độ quan hệ nhân quả — Chương 4.4.

    Parameters
    ----------
    weight_matrix : np.ndarray (d, d)   ma trận W raw từ NOTEARS.
    node_names    : List[str]
    title         : str
    output_path   : str, optional
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("pip install matplotlib seaborn")

    labels = [_LABELS.get(n, n) for n in node_names]
    vmax   = float(np.abs(weight_matrix).max())

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        weight_matrix,
        annot       = True,
        fmt         = ".3f",
        cmap        = "RdBu_r",
        center      = 0,
        vmin        = -vmax,
        vmax        = vmax,
        xticklabels = labels,
        yticklabels = labels,
        linewidths  = 0.5,
        ax          = ax,
        annot_kws   = {"size": 10},
    )
    ax.set_xlabel("Target (nhận rủi ro)", fontsize=11)
    ax.set_ylabel("Source (phát rủi ro)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    _save_fig(fig, output_path)
    return fig


# ─── Internal helper ─────────────────────────────────────────────────────────

def _save_fig(fig: "plt.Figure", path: Optional[str]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        logger.info(f"Đã lưu: {path}")
