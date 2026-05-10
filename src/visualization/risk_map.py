"""
src/visualization/risk_map.py
Causal Risk Map — Figure chính Chương 4.6.

Đặt file tại: src/visualization/risk_map.py
"""

import numpy as np
import logging
from typing import List, Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

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
# 1. Risk Map tổng hợp 3 phương pháp — Figure chính Chương 4.6
# ─────────────────────────────────────────────────────────────────────────────

def plot_risk_map(
    results    : Dict[str, object],
    node_names : List[str],
    output_path: Optional[str] = None,
    figsize    : tuple = (18, 6),
) -> "plt.Figure":
    """
    Causal Risk Map tổng hợp: 3 subplots cho 3 phương pháp trên cùng 1 figure.

    Node size ∝ out-degree → trực quan hóa "ai phát rủi ro nhiều nhất".

    Parameters
    ----------
    results : dict
        {"Granger": CausalResult, "PC Algorithm": CausalResult, "NOTEARS": CausalResult}
        Thứ tự trong dict = thứ tự subplot (trái → phải).
    node_names : List[str]
        Danh sách thị trường.
    output_path : str, optional
        Đường dẫn lưu ảnh.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> fig = plot_risk_map(
    ...     results    = {"Granger": granger_r, "PC Algorithm": pc_r, "NOTEARS": notears_r},
    ...     node_names = ["SP500","VNINDEX","Gold","WTI_Oil","Bitcoin"],
    ...     output_path= "reports/figures/causal_risk_map.png",
    ... )
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        raise ImportError("pip install matplotlib networkx")

    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]

    for ax, (method_name, result) in zip(axes, results.items()):
        G = nx.DiGraph()
        G.add_nodes_from(node_names)
        for i, src in enumerate(node_names):
            for j, tgt in enumerate(node_names):
                if result.binary_matrix[i, j] == 1:
                    G.add_edge(src, tgt)

        pos         = nx.circular_layout(G)
        out_deg     = dict(G.out_degree())
        in_deg      = dict(G.in_degree())
        node_colors = [_COLORS.get(n, "#aaa") for n in G.nodes()]
        node_sizes  = [700 + out_deg.get(n, 0) * 320 for n in G.nodes()]

        nx.draw_networkx_nodes(
            G, pos,
            node_color = node_colors,
            node_size  = node_sizes,
            alpha      = 0.88,
            ax         = ax,
        )
        nx.draw_networkx_labels(
            G, pos,
            labels      = {n: _LABELS.get(n, n) for n in G.nodes()},
            font_size   = 9,
            font_weight = "bold",
            ax          = ax,
        )
        if G.number_of_edges() > 0:
            nx.draw_networkx_edges(
                G, pos,
                arrows          = True,
                arrowsize       = 18,
                edge_color      = "#444444",
                width           = 2.0,
                connectionstyle = "arc3,rad=0.1",
                ax              = ax,
                min_source_margin = 18,
                min_target_margin = 18,
            )

        # Đánh dấu ngôi sao cho NOTEARS
        star  = "★ " if method_name.upper() == "NOTEARS" else ""
        ax.set_title(
            f"{star}{method_name}\n({result.n_edges} cạnh)",
            fontsize   = 11,
            fontweight = "bold",
            pad        = 10,
        )

        # Ghi chú hub node (out-degree cao nhất)
        if out_deg:
            hub = max(out_deg, key=out_deg.get)
            ax.text(
                0.5, -0.04,
                f"Hub: {_LABELS.get(hub, hub)} (out={out_deg[hub]})",
                transform  = ax.transAxes,
                ha         = "center",
                fontsize   = 8,
                color      = "dimgray",
                fontstyle  = "italic",
            )

        ax.axis("off")

    fig.suptitle(
        "Causal Risk Map — So sánh 3 Phương pháp Causal Discovery\n"
        "Rolling Volatility 5 Thị trường Tài chính (2018–2024)",
        fontsize   = 13,
        fontweight = "bold",
        y          = 1.02,
    )
    plt.tight_layout()
    _save_fig(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Degree Analysis Bar Chart — Chương 4.4
# ─────────────────────────────────────────────────────────────────────────────

def plot_degree_analysis(
    result     : object,
    node_names : List[str],
    title      : str = "In-degree vs Out-degree",
    output_path: Optional[str] = None,
    figsize    : tuple = (10, 5),
) -> "plt.Figure":
    """
    Grouped bar chart: out-degree vs in-degree cho từng thị trường.

    Diễn giải:
      - Out-degree cao → nguồn PHÁT rủi ro (risk emitter)
      - In-degree cao  → nơi NHẬN rủi ro   (risk receiver)
      - Net = out - in → dương: emitter, âm: receiver

    Parameters
    ----------
    result     : CausalResult   kết quả từ một model (thường là NOTEARS)
    node_names : List[str]
    title      : str
    output_path: str, optional
    figsize    : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("pip install matplotlib")

    out_deg = result.out_degree().astype(int)
    in_deg  = result.in_degree().astype(int)
    net     = out_deg - in_deg
    labels  = [_LABELS.get(n, n) for n in node_names]
    colors  = [_COLORS.get(n, "#aaa") for n in node_names]
    x       = np.arange(len(node_names))
    width   = 0.30

    fig, (ax_main, ax_net) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ── Main: grouped bar ────────────────────────────────────────────────────
    bars_out = ax_main.bar(
        x - width / 2, out_deg, width,
        label     = "Out-degree (phát rủi ro)",
        color     = colors,
        alpha     = 0.87,
        edgecolor = "white",
    )
    bars_in = ax_main.bar(
        x + width / 2, in_deg, width,
        label     = "In-degree (nhận rủi ro)",
        color     = colors,
        alpha     = 0.42,
        hatch     = "///",
        edgecolor = "gray",
    )

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(labels, fontsize=11)
    ax_main.set_ylabel("Degree", fontsize=11)
    ax_main.set_title(title, fontsize=13, fontweight="bold")
    ax_main.legend(fontsize=10)
    ax_main.grid(axis="y", alpha=0.3, linestyle="--")
    ax_main.bar_label(bars_out, padding=2, fontsize=9)
    ax_main.bar_label(bars_in,  padding=2, fontsize=9)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    # ── Net degree bar ────────────────────────────────────────────────────────
    net_colors = ["#E63946" if v > 0 else "#118AB2" for v in net]
    ax_net.bar(x, net, color=net_colors, alpha=0.8, edgecolor="white")
    ax_net.axhline(0, color="black", linewidth=0.8)
    ax_net.set_xticks(x)
    ax_net.set_xticklabels(labels, fontsize=9)
    ax_net.set_ylabel("Net\n(out−in)", fontsize=9)
    ax_net.set_title("Net degree (đỏ = emitter, xanh = receiver)",
                     fontsize=9, color="dimgray")
    ax_net.grid(axis="y", alpha=0.25, linestyle="--")
    ax_net.spines["top"].set_visible(False)
    ax_net.spines["right"].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Edge Stability Heatmap — Chương 4.6 (subperiod)
# ─────────────────────────────────────────────────────────────────────────────

def plot_edge_stability(
    freq_matrix: np.ndarray,
    node_names : List[str],
    title      : str = "Độ ổn định cạnh nhân quả qua các giai đoạn",
    output_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Heatmap tần suất xuất hiện của từng cạnh qua các giai đoạn.

    freq_matrix[i,j] = tỷ lệ giai đoạn mà cạnh i→j tồn tại (0–1).
    Cạnh tần suất cao = quan hệ nhân quả bền vững.

    Parameters
    ----------
    freq_matrix : np.ndarray (d, d)   giá trị trong [0, 1]
    node_names  : List[str]
    title       : str
    output_path : str, optional
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("pip install matplotlib seaborn")

    labels = [_LABELS.get(n, n) for n in node_names]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        freq_matrix,
        annot       = True,
        fmt         = ".2f",
        cmap        = "YlOrRd",
        vmin        = 0,
        vmax        = 1,
        xticklabels = labels,
        yticklabels = labels,
        linewidths  = 0.5,
        ax          = ax,
        annot_kws   = {"size": 11},
    )
    ax.set_xlabel("Target (nhận rủi ro)", fontsize=11)
    ax.set_ylabel("Source (phát rủi ro)", fontsize=11)
    ax.set_title(
        f"{title}\n"
        "(0 = không bao giờ xuất hiện  |  1 = luôn xuất hiện)",
        fontsize   = 11,
        fontweight = "bold",
        pad        = 12,
    )
    plt.tight_layout()
    _save_fig(fig, output_path)
    return fig


# ─── Internal helper ─────────────────────────────────────────────────────────

def _save_fig(fig: "plt.Figure", path: Optional[str]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        logger.info(f"Đã lưu: {path}")
