from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CAMERA_READY_DIR
from src.camera_ready_analysis import DISPLAY_LABELS, graph_node_metrics
from src.camera_ready_models import sparsify_importance


QUANTILES = (0.60, 0.70, 0.80)
TABLE_DIR = CAMERA_READY_DIR / "tables"
STABILITY_DIR = TABLE_DIR / "stability_runs"


def load_saved_seed_importance(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    match = re.fullmatch(r"seed_(\d+)_importance\.csv", path.name)
    if not match:
        raise ValueError(f"Unexpected stability filename: {path.name}")
    seed = int(match.group(1))
    frame = pd.read_csv(path)
    required = {
        "source",
        "target",
        "relative_loss_increase",
        "signed_relative_loss_change",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")

    positive = np.zeros((len(DISPLAY_LABELS), len(DISPLAY_LABELS)), dtype=float)
    signed = np.zeros_like(positive)
    seen: set[tuple[str, str]] = set()
    for row in frame.itertuples(index=False):
        if row.source == row.target:
            continue
        source = DISPLAY_LABELS.index(row.source)
        target = DISPLAY_LABELS.index(row.target)
        key = (row.source, row.target)
        if key in seen:
            raise ValueError(f"Duplicate edge in {path}: {key}")
        seen.add(key)
        positive[source, target] = max(float(row.relative_loss_increase), 0.0)
        signed[source, target] = float(row.signed_relative_loss_change)
    expected = len(DISPLAY_LABELS) * (len(DISPLAY_LABELS) - 1)
    if len(seen) != expected:
        raise ValueError(f"Expected {expected} directed off-diagonal pairs in {path}, got {len(seen)}")
    return seed, positive, signed


def load_all_saved_matrices(
    stability_dir: Path = STABILITY_DIR,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    matrices: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(stability_dir.glob("seed_*_importance.csv")):
        seed, positive, signed = load_saved_seed_importance(path)
        matrices[seed] = (positive, signed)
    if not matrices:
        raise FileNotFoundError(f"No saved seed importance files found in {stability_dir}")
    return dict(sorted(matrices.items()))


def graph_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    mask = ~np.eye(left.shape[0], dtype=bool)
    left_edges = (left > 0) & mask
    right_edges = (right > 0) & mask
    union = int(np.sum(left_edges | right_edges))
    return float(np.sum(left_edges & right_edges) / union) if union else 1.0


def analyze_saved_matrices(
    matrices: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    graph_summary_rows: list[dict] = []
    node_rows: list[dict] = []
    retained_by_key: dict[tuple[int, float], np.ndarray] = {}
    threshold_by_key: dict[tuple[int, float], float] = {}

    for seed, (positive, _) in matrices.items():
        for quantile in QUANTILES:
            retained, threshold = sparsify_importance(positive, quantile)
            retained_by_key[(seed, quantile)] = retained
            threshold_by_key[(seed, quantile)] = threshold
            nodes, graph = graph_node_metrics(retained)
            nodes = nodes.set_index("node")
            vni = nodes.loc["VNI"]
            incoming_vni = sorted(graph.predecessors("VNI"))
            graph_summary_rows.append(
                {
                    "quantile": quantile,
                    "seed": seed,
                    "threshold": threshold,
                    "edge_count": int(np.count_nonzero(retained)),
                    "incoming_vni_sources": ";".join(incoming_vni),
                    "vni_in_strength": float(vni["in_strength"]),
                    "vni_out_strength": float(vni["out_strength"]),
                    "vni_net_strength": float(vni["net_strength"]),
                    "vni_role": vni["role"],
                }
            )
            for node, row in nodes.iterrows():
                node_rows.append(
                    {
                        "quantile": quantile,
                        "seed": seed,
                        "node": node,
                        "in_degree": int(row["in_degree"]),
                        "out_degree": int(row["out_degree"]),
                        "in_strength": float(row["in_strength"]),
                        "out_strength": float(row["out_strength"]),
                        "net_strength": float(row["net_strength"]),
                        "role": row["role"],
                    }
                )

    graph_summary = pd.DataFrame(graph_summary_rows).sort_values(["quantile", "seed"])
    node_roles = pd.DataFrame(node_rows).sort_values(["quantile", "seed", "node"])
    q70_roles = node_roles.loc[node_roles["quantile"].eq(0.70), ["seed", "node", "role"]].rename(
        columns={"role": "q70_role"}
    )
    node_roles = node_roles.merge(q70_roles, on=["seed", "node"], how="left")
    node_roles["agrees_with_q70_role"] = node_roles["role"].eq(node_roles["q70_role"])

    edge_rows: list[dict] = []
    for quantile in QUANTILES:
        selected = np.stack(
            [(retained_by_key[(seed, quantile)] > 0) for seed in matrices]
        )
        positives = np.stack([matrices[seed][0] for seed in matrices])
        signed = np.stack([matrices[seed][1] for seed in matrices])
        seed42_retained = retained_by_key.get((42, quantile))
        for source, source_label in enumerate(DISPLAY_LABELS):
            for target, target_label in enumerate(DISPLAY_LABELS):
                if source == target:
                    continue
                edge_rows.append(
                    {
                        "quantile": quantile,
                        "source": source_label,
                        "target": target_label,
                        "selected_seeds": int(selected[:, source, target].sum()),
                        "available_seeds": len(matrices),
                        "selection_frequency": float(selected[:, source, target].mean()),
                        "mean_positive_importance": float(positives[:, source, target].mean()),
                        "std_positive_importance": float(positives[:, source, target].std(ddof=1)),
                        "mean_signed_loss_change": float(signed[:, source, target].mean()),
                        "std_signed_loss_change": float(signed[:, source, target].std(ddof=1)),
                        "seed42_retained": bool(
                            seed42_retained is not None
                            and seed42_retained[source, target] > 0
                        ),
                    }
                )
    edge_frequency = pd.DataFrame(edge_rows).sort_values(
        ["quantile", "selection_frequency", "mean_positive_importance"],
        ascending=[True, False, False],
    )

    jaccard_rows: list[dict] = []
    for seed in matrices:
        q70 = retained_by_key[(seed, 0.70)]
        seed_roles = node_roles.loc[node_roles["seed"].eq(seed)]
        for quantile in QUANTILES:
            roles = seed_roles.loc[seed_roles["quantile"].eq(quantile)]
            jaccard_rows.append(
                {
                    "quantile": quantile,
                    "seed": seed,
                    "edge_jaccard_with_q70": graph_jaccard(
                        retained_by_key[(seed, quantile)], q70
                    ),
                    "node_role_agreement_with_q70": float(
                        roles["agrees_with_q70_role"].mean()
                    ),
                    "vni_role_agreement_with_q70": bool(
                        roles.loc[roles["node"].eq("VNI"), "agrees_with_q70_role"].iloc[0]
                    ),
                    "edge_count": int(np.count_nonzero(retained_by_key[(seed, quantile)])),
                    "q70_edge_count": int(np.count_nonzero(q70)),
                }
            )
    jaccard = pd.DataFrame(jaccard_rows).sort_values(["quantile", "seed"])
    diagnostics = {
        "retained": retained_by_key,
        "thresholds": threshold_by_key,
    }
    return graph_summary, edge_frequency, node_roles, jaccard, diagnostics


def verify_q70_seed42(
    diagnostics: dict,
    table_dir: Path = TABLE_DIR,
) -> None:
    retained = diagnostics["retained"][(42, 0.70)]
    threshold = diagnostics["thresholds"][(42, 0.70)]
    canonical = pd.read_csv(table_dir / "ablation_importance_retained_seed42.csv")
    canonical_matrix = canonical.set_index("source").loc[DISPLAY_LABELS, DISPLAY_LABELS].to_numpy(float)
    expected_threshold = float(
        pd.read_csv(table_dir / "stability_run_summary.csv")
        .set_index("seed")
        .loc[42, "q70_threshold"]
    )
    if int(np.count_nonzero(retained)) != 7:
        raise AssertionError("q0.70 seed-42 graph does not contain the canonical seven edges")
    np.testing.assert_allclose(retained, canonical_matrix, rtol=0.0, atol=1e-15)
    if not np.isclose(threshold, expected_threshold, rtol=0.0, atol=1e-15):
        raise AssertionError(
            f"q0.70 threshold mismatch: reconstructed={threshold}, canonical={expected_threshold}"
        )


def verify_monotonic_edge_counts(graph_summary: pd.DataFrame) -> None:
    for seed, frame in graph_summary.groupby("seed"):
        counts = frame.set_index("quantile").loc[list(QUANTILES), "edge_count"].to_numpy()
        if np.any(np.diff(counts) > 0):
            raise AssertionError(f"Higher quantile retained more edges for seed {seed}: {counts}")


def draw_edge_frequency_figure(edge_frequency: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    image = None
    for ax, quantile in zip(axes, QUANTILES):
        frame = edge_frequency.loc[edge_frequency["quantile"].eq(quantile)]
        matrix = np.zeros((len(DISPLAY_LABELS), len(DISPLAY_LABELS)))
        for row in frame.itertuples(index=False):
            matrix[DISPLAY_LABELS.index(row.source), DISPLAY_LABELS.index(row.target)] = row.selection_frequency
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"q{int(quantile * 100)}")
        ax.set_xticks(range(len(DISPLAY_LABELS)), DISPLAY_LABELS, rotation=45, ha="right")
        ax.set_yticks(range(len(DISPLAY_LABELS)), DISPLAY_LABELS)
        ax.set_xlabel("Target")
        if ax is axes[0]:
            ax.set_ylabel("Source")
    fig.colorbar(image, ax=axes, label="Selection frequency", shrink=0.85)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_vni_sensitivity_figure(graph_summary: pd.DataFrame, path: Path) -> None:
    grouped = graph_summary.groupby("quantile")
    summary = grouped.agg(mean_edges=("edge_count", "mean")).reset_index()
    roles = (
        graph_summary.assign(
            transmitter=graph_summary["vni_role"].eq("Transmitter").astype(float),
            receiver=graph_summary["vni_role"].eq("Receiver").astype(float),
        )
        .groupby("quantile")
        .agg(
            vni_transmitter_frequency=("transmitter", "mean"),
            vni_receiver_frequency=("receiver", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(roles, on="quantile")
    x = [f"q{int(value * 100)}" for value in summary["quantile"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].plot(x, summary["mean_edges"], marker="o", color="#355C7D")
    axes[0].set_ylabel("Mean retained edges")
    axes[0].set_title("Graph density across 20 seeds")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, summary["vni_transmitter_frequency"], marker="o", label="Transmitter")
    axes[1].plot(x, summary["vni_receiver_frequency"], marker="s", label="Receiver")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Frequency across seeds")
    axes[1].set_title("VNI role sensitivity")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_report(
    graph_summary: pd.DataFrame,
    edge_frequency: pd.DataFrame,
    node_roles: pd.DataFrame,
    jaccard: pd.DataFrame,
) -> str:
    seed42 = graph_summary.loc[graph_summary["seed"].eq(42)].copy()
    aggregate = (
        graph_summary.assign(
            vni_transmitter=graph_summary["vni_role"].eq("Transmitter").astype(float),
            vni_receiver=graph_summary["vni_role"].eq("Receiver").astype(float),
        )
        .groupby("quantile")
        .agg(
            mean_edge_count=("edge_count", "mean"),
            min_edge_count=("edge_count", "min"),
            max_edge_count=("edge_count", "max"),
            vni_transmitter_frequency=("vni_transmitter", "mean"),
            vni_receiver_frequency=("vni_receiver", "mean"),
            mean_vni_net_strength=("vni_net_strength", "mean"),
        )
        .reset_index()
    )
    overlap = (
        jaccard.groupby("quantile")
        .agg(
            mean_edge_jaccard_with_q70=("edge_jaccard_with_q70", "mean"),
            mean_node_role_agreement=("node_role_agreement_with_q70", "mean"),
            vni_role_agreement_frequency=("vni_role_agreement_with_q70", "mean"),
        )
        .reset_index()
    )
    top_edges = (
        edge_frequency.sort_values(
            ["quantile", "selection_frequency", "mean_positive_importance"],
            ascending=[True, False, False],
        )
        .groupby("quantile", as_index=False)
        .head(5)
    )
    role_agreement = (
        node_roles.groupby(["quantile", "node"])
        .agg(agreement_with_q70=("agrees_with_q70_role", "mean"))
        .reset_index()
    )
    return "\n".join(
        [
            "# Graph Threshold Sensitivity",
            "",
            "## Outcome",
            "",
            "Saved positive and signed block-ablation outputs for all available stability seeds were re-thresholded at the pre-specified q0.60, q0.70 and q0.80 levels. No model was retrained. q0.70 remains the camera-ready primary specification.",
            "",
            "## Seed-42 graphs",
            "",
            seed42.to_markdown(index=False),
            "",
            "## Across-seed graph and VNI sensitivity",
            "",
            aggregate.to_markdown(index=False),
            "",
            "## Agreement with q0.70",
            "",
            overlap.to_markdown(index=False),
            "",
            "## Five most frequently selected edges at each threshold",
            "",
            top_edges[["quantile", "source", "target", "selection_frequency", "mean_positive_importance", "seed42_retained"]].to_markdown(index=False),
            "",
            "## Node-role agreement with q0.70",
            "",
            role_agreement.to_markdown(index=False),
            "",
            "## Interpretation boundary",
            "",
            "- q0.60 and q0.80 are fixed sensitivity specifications, not alternatives selected after viewing the results.",
            "- Differences in density, edge frequency or VNI role indicate threshold sensitivity and must not be hidden.",
            "- Edge selection is predictive block-ablation importance, not structural causality.",
            "- The camera-ready q0.70 graph and all existing camera-ready outputs remain unchanged.",
        ]
    )


def run_analysis(output_dir: Path) -> dict[str, object]:
    matrices = load_all_saved_matrices()
    graph_summary, edge_frequency, node_roles, jaccard, diagnostics = analyze_saved_matrices(matrices)
    verify_q70_seed42(diagnostics)
    verify_monotonic_edge_counts(graph_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    graph_summary.to_csv(output_dir / "threshold_graph_summary.csv", index=False)
    edge_frequency.to_csv(output_dir / "threshold_edge_frequency.csv", index=False)
    node_roles.to_csv(output_dir / "threshold_node_roles.csv", index=False)
    jaccard.to_csv(output_dir / "threshold_jaccard.csv", index=False)
    (output_dir / "graph_sensitivity.md").write_text(
        render_report(graph_summary, edge_frequency, node_roles, jaccard),
        encoding="utf-8",
    )
    draw_edge_frequency_figure(edge_frequency, output_dir / "threshold_edge_frequency.png")
    draw_vni_sensitivity_figure(graph_summary, output_dir / "threshold_vni_sensitivity.png")
    return {
        "seeds": list(matrices),
        "graph_summary": graph_summary,
        "edge_frequency": edge_frequency,
        "node_roles": node_roles,
        "jaccard": jaccard,
        "diagnostics": diagnostics,
    }
