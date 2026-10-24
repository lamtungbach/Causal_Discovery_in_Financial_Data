from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CAMERA_READY_DIR
from src.camera_ready_analysis import DISPLAY_LABELS, graph_node_metrics


TABLE_DIR = CAMERA_READY_DIR / "tables"
STABILITY_DIR = TABLE_DIR / "stability_runs"
CONSENSUS_LEVELS = (0.50, 0.65, 0.80)
WILSON_Z_95 = 1.959963984540054


def _parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def load_saved_seed_graphs(
    stability_dir: Path = STABILITY_DIR,
) -> dict[int, dict[str, np.ndarray]]:
    graphs: dict[int, dict[str, np.ndarray]] = {}
    expected_edges = len(DISPLAY_LABELS) * (len(DISPLAY_LABELS) - 1)
    for path in sorted(stability_dir.glob("seed_*_importance.csv")):
        match = re.fullmatch(r"seed_(\d+)_importance\.csv", path.name)
        if not match:
            continue
        seed = int(match.group(1))
        frame = pd.read_csv(path)
        required = {
            "source",
            "target",
            "relative_loss_increase",
            "signed_relative_loss_change",
            "retained",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks required columns: {sorted(missing)}")

        positive = np.zeros((len(DISPLAY_LABELS), len(DISPLAY_LABELS)), dtype=float)
        signed = np.zeros_like(positive)
        selected = np.zeros_like(positive, dtype=bool)
        seen: set[tuple[str, str]] = set()
        for row in frame.itertuples(index=False):
            if row.source == row.target:
                continue
            key = (row.source, row.target)
            if key in seen:
                raise ValueError(f"Duplicate edge in {path}: {key}")
            seen.add(key)
            source = DISPLAY_LABELS.index(row.source)
            target = DISPLAY_LABELS.index(row.target)
            positive[source, target] = max(float(row.relative_loss_increase), 0.0)
            signed[source, target] = float(row.signed_relative_loss_change)
            selected[source, target] = _parse_bool(row.retained)
        if len(seen) != expected_edges:
            raise ValueError(f"Expected {expected_edges} edges in {path}, got {len(seen)}")
        if np.any(selected & (positive <= 0)):
            raise ValueError(f"Retained non-positive edge found in {path}")
        graphs[seed] = {"positive": positive, "signed": signed, "selected": selected}
    if not graphs:
        raise FileNotFoundError(f"No seed importance files found in {stability_dir}")
    return dict(sorted(graphs.items()))


def wilson_interval(successes: int, trials: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie between zero and trials")
    p = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (p + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(p * (1.0 - p) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def edge_frequency_intervals(graphs: dict[int, dict[str, np.ndarray]]) -> pd.DataFrame:
    selected = np.stack([item["selected"] for item in graphs.values()])
    positive = np.stack([item["positive"] for item in graphs.values()])
    signed = np.stack([item["signed"] for item in graphs.values()])
    n = len(graphs)
    rows: list[dict] = []
    for source, source_label in enumerate(DISPLAY_LABELS):
        for target, target_label in enumerate(DISPLAY_LABELS):
            if source == target:
                continue
            count = int(selected[:, source, target].sum())
            lower, upper = wilson_interval(count, n)
            rows.append(
                {
                    "source": source_label,
                    "target": target_label,
                    "selected_seeds": count,
                    "available_seeds": n,
                    "selection_frequency": count / n,
                    "wilson_95_lower": lower,
                    "wilson_95_upper": upper,
                    "mean_positive_importance": float(positive[:, source, target].mean()),
                    "std_positive_importance": float(positive[:, source, target].std(ddof=1)),
                    "mean_signed_loss_change": float(signed[:, source, target].mean()),
                    "std_signed_loss_change": float(signed[:, source, target].std(ddof=1)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["selection_frequency", "mean_positive_importance"], ascending=False
    )


def node_role_outputs(
    graphs: dict[int, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_rows: list[dict] = []
    for seed, item in graphs.items():
        retained_weights = np.where(item["selected"], item["positive"], 0.0)
        nodes, _ = graph_node_metrics(retained_weights)
        for row in nodes.itertuples(index=False):
            per_seed_rows.append(
                {
                    "seed": seed,
                    "node": row.node,
                    "in_strength": float(row.in_strength),
                    "out_strength": float(row.out_strength),
                    "net_strength": float(row.net_strength),
                    "role": row.role,
                }
            )
    per_seed = pd.DataFrame(per_seed_rows).sort_values(["seed", "node"])

    interval_rows: list[dict] = []
    n = len(graphs)
    for node in DISPLAY_LABELS:
        node_frame = per_seed.loc[per_seed["node"].eq(node)]
        for role in ("Transmitter", "Receiver", "Balanced"):
            count = int(node_frame["role"].eq(role).sum())
            lower, upper = wilson_interval(count, n)
            interval_rows.append(
                {
                    "node": node,
                    "role": role,
                    "role_seeds": count,
                    "available_seeds": n,
                    "role_frequency": count / n,
                    "wilson_95_lower": lower,
                    "wilson_95_upper": upper,
                }
            )
    intervals = pd.DataFrame(interval_rows).sort_values(["node", "role"])
    vni = per_seed.loc[per_seed["node"].eq("VNI")].reset_index(drop=True)
    return intervals, vni


def binary_jaccard(left: np.ndarray, right: np.ndarray) -> tuple[float, int, int, int]:
    mask = ~np.eye(left.shape[0], dtype=bool)
    left = left.astype(bool) & mask
    right = right.astype(bool) & mask
    intersection = int(np.sum(left & right))
    union = int(np.sum(left | right))
    turnover = int(np.sum(left ^ right))
    jaccard = intersection / union if union else 1.0
    return float(jaccard), intersection, union, turnover


def pairwise_jaccard_table(graphs: dict[int, dict[str, np.ndarray]]) -> pd.DataFrame:
    seeds = list(graphs)
    rows: list[dict] = []
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            jaccard, intersection, union, turnover = binary_jaccard(
                graphs[seed_a]["selected"], graphs[seed_b]["selected"]
            )
            rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "includes_seed42": seed_a == 42 or seed_b == 42,
                    "intersection_edges": intersection,
                    "union_edges": union,
                    "turnover_edges": turnover,
                    "edge_jaccard": jaccard,
                    "turnover_rate": 1.0 - jaccard,
                }
            )
    return pd.DataFrame(rows).sort_values(["seed_a", "seed_b"])


def consensus_graph_table(edge_intervals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for level in CONSENSUS_LEVELS:
        for row in edge_intervals.itertuples(index=False):
            rows.append(
                {
                    "consensus_level": level,
                    "source": row.source,
                    "target": row.target,
                    "selection_frequency": row.selection_frequency,
                    "selected_in_consensus": bool(row.selection_frequency >= level),
                    "mean_positive_importance": row.mean_positive_importance,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["consensus_level", "selected_in_consensus", "selection_frequency"],
        ascending=[True, False, False],
    )


def verify_against_canonical(
    edge_intervals: pd.DataFrame,
    role_intervals: pd.DataFrame,
    consensus: pd.DataFrame,
    table_dir: Path = TABLE_DIR,
) -> None:
    canonical_edges = pd.read_csv(table_dir / "edge_stability_20seeds.csv")
    merged = canonical_edges[["source", "target", "selection_frequency", "stable"]].merge(
        edge_intervals[["source", "target", "selection_frequency"]],
        on=["source", "target"],
        suffixes=("_canonical", "_reconstructed"),
        validate="one_to_one",
    )
    np.testing.assert_allclose(
        merged["selection_frequency_canonical"],
        merged["selection_frequency_reconstructed"],
        rtol=0.0,
        atol=0.0,
    )

    canonical_roles = pd.read_csv(table_dir / "node_role_stability_20seeds.csv")
    for role, column in (("Transmitter", "transmitter_frequency"), ("Receiver", "receiver_frequency")):
        reconstructed = role_intervals.loc[role_intervals["role"].eq(role), ["node", "role_frequency"]]
        role_check = canonical_roles[["node", column]].merge(
            reconstructed, on="node", validate="one_to_one"
        )
        np.testing.assert_allclose(role_check[column], role_check["role_frequency"], rtol=0.0, atol=0.0)

    selected_80 = consensus.loc[
        consensus["consensus_level"].eq(0.80) & consensus["selected_in_consensus"]
    ][["source", "target"]]
    canonical_stable = canonical_edges.loc[canonical_edges["stable"].map(_parse_bool), ["source", "target"]]
    actual = set(map(tuple, selected_80.to_numpy()))
    expected = set(map(tuple, canonical_stable.to_numpy()))
    if actual != expected:
        raise AssertionError(f"0.80 consensus mismatch: actual={actual}, expected={expected}")


def draw_edge_interval_figure(edge_intervals: pd.DataFrame, path: Path) -> None:
    visible = edge_intervals.loc[edge_intervals["selected_seeds"].gt(0)].copy()
    visible = visible.sort_values(["selection_frequency", "mean_positive_importance"]).reset_index(drop=True)
    labels = visible["source"] + " -> " + visible["target"]
    y = np.arange(len(visible))
    lower = visible["selection_frequency"] - visible["wilson_95_lower"]
    upper = visible["wilson_95_upper"] - visible["selection_frequency"]
    fig_height = max(5.0, 0.28 * len(visible) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.errorbar(
        visible["selection_frequency"],
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#355C7D",
        ecolor="#8AA1B4",
        capsize=3,
    )
    ax.axvline(0.80, color="#C44E52", linestyle="--", linewidth=1.2, label="0.80 consensus")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Selection frequency with 95% Wilson interval")
    ax.set_title("Cross-seed edge-selection variability (20 seeds)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_vni_distribution(vni: pd.DataFrame, path: Path) -> None:
    ordered = vni.sort_values("seed")
    colors = ordered["role"].map(
        {"Transmitter": "#3B82A0", "Receiver": "#D97745", "Balanced": "#8A8A8A"}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].scatter(ordered["seed"], ordered["net_strength"], c=colors, s=45)
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("VNI net strength")
    axes[0].set_title("VNI net strength by seed")
    axes[0].grid(alpha=0.25)
    axes[1].hist(ordered["net_strength"], bins=8, color="#6B8EAD", edgecolor="white")
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("VNI net strength")
    axes[1].set_ylabel("Seeds")
    axes[1].set_title("Distribution across 20 seeds")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_report(
    edge_intervals: pd.DataFrame,
    role_intervals: pd.DataFrame,
    pairwise: pd.DataFrame,
    consensus: pd.DataFrame,
    vni: pd.DataFrame,
    reproducibility: dict,
) -> str:
    selected_consensus = consensus.loc[consensus["selected_in_consensus"]].copy()
    consensus_counts = (
        selected_consensus.groupby("consensus_level")
        .size()
        .reindex(CONSENSUS_LEVELS, fill_value=0)
        .rename("edge_count")
        .reset_index()
    )
    consensus_edges = selected_consensus.assign(
        edge=selected_consensus["source"] + " -> " + selected_consensus["target"]
    ).groupby("consensus_level")["edge"].apply(lambda values: "; ".join(values)).reindex(CONSENSUS_LEVELS, fill_value="").reset_index(name="edges")
    consensus_summary = consensus_counts.merge(consensus_edges, on="consensus_level")
    vni_roles = role_intervals.loc[role_intervals["node"].eq("VNI")].copy()
    all_pairs = pairwise["edge_jaccard"]
    seed42_pairs = pairwise.loc[pairwise["includes_seed42"], "edge_jaccard"]
    turnover_summary = pd.DataFrame(
        [
            {
                "comparison": "all_seed_pairs",
                "pairs": len(all_pairs),
                "mean_jaccard": all_pairs.mean(),
                "median_jaccard": all_pairs.median(),
                "min_jaccard": all_pairs.min(),
                "max_jaccard": all_pairs.max(),
                "mean_turnover_rate": 1.0 - all_pairs.mean(),
            },
            {
                "comparison": "seed42_vs_other",
                "pairs": len(seed42_pairs),
                "mean_jaccard": seed42_pairs.mean(),
                "median_jaccard": seed42_pairs.median(),
                "min_jaccard": seed42_pairs.min(),
                "max_jaccard": seed42_pairs.max(),
                "mean_turnover_rate": 1.0 - seed42_pairs.mean(),
            },
        ]
    )
    top_edges = edge_intervals.head(12)[
        ["source", "target", "selected_seeds", "selection_frequency", "wilson_95_lower", "wilson_95_upper"]
    ]
    return "\n".join(
        [
            "# Seed-Based Uncertainty",
            "",
            "## Outcome",
            "",
            "The frozen q0.70 retained graphs and ablation outputs for seeds 42-61 were analyzed without retraining. Intervals below describe finite 20-seed selection proportions; they are not confidence intervals for a market population or for sampling uncertainty.",
            "",
            "## Most frequently selected edges",
            "",
            top_edges.to_markdown(index=False),
            "",
            "## Consensus graphs",
            "",
            consensus_summary.to_markdown(index=False),
            "",
            "## VNI role frequencies",
            "",
            vni_roles.to_markdown(index=False),
            "",
            "## Edge turnover and Jaccard overlap",
            "",
            turnover_summary.to_markdown(index=False),
            "",
            "## VNI net-strength distribution",
            "",
            pd.DataFrame(
                [
                    {
                        "seeds": len(vni),
                        "mean": vni["net_strength"].mean(),
                        "std": vni["net_strength"].std(ddof=1),
                        "min": vni["net_strength"].min(),
                        "median": vni["net_strength"].median(),
                        "max": vni["net_strength"].max(),
                    }
                ]
            ).to_markdown(index=False),
            "",
            "## Reproducibility versus cross-seed variability",
            "",
            f"- Saved deterministic same-seed check (seed {reproducibility.get('seed_repeated')}): maximum absolute repeat difference `{reproducibility.get('max_abs_repeat_diff')}`.",
            f"- Saved different-seed comparison (seed {reproducibility.get('comparison_seed')}): maximum absolute prediction difference `{reproducibility.get('max_abs_different_seed_diff')}`.",
            "- The first item concerns deterministic repeatability under the same seed; the tables in this Part concern variation across different optimization seeds. These are distinct concepts.",
            "",
            "## Interpretation boundary",
            "",
            "- An edge below the consensus cutoffs remains unstable; it is not relabeled as a confirmed spillover.",
            "- Wilson intervals quantify uncertainty around the observed 20-seed proportions only.",
            "- Edge presence represents predictive block-ablation importance, not structural causality.",
            "- No model, threshold, camera-ready output or manuscript was changed.",
        ]
    )


def run_analysis(output_dir: Path) -> dict[str, object]:
    graphs = load_saved_seed_graphs()
    edge_intervals = edge_frequency_intervals(graphs)
    role_intervals, vni = node_role_outputs(graphs)
    pairwise = pairwise_jaccard_table(graphs)
    consensus = consensus_graph_table(edge_intervals)
    verify_against_canonical(edge_intervals, role_intervals, consensus)

    reproducibility = json.loads(
        (TABLE_DIR / "reproducibility_check.json").read_text(encoding="utf-8")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_intervals.to_csv(output_dir / "edge_frequency_intervals.csv", index=False)
    role_intervals.to_csv(output_dir / "node_role_frequency_intervals.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_jaccard.csv", index=False)
    consensus.to_csv(output_dir / "consensus_graphs.csv", index=False)
    vni.to_csv(output_dir / "vni_net_strength_distribution.csv", index=False)
    (output_dir / "seed_uncertainty.md").write_text(
        render_report(edge_intervals, role_intervals, pairwise, consensus, vni, reproducibility),
        encoding="utf-8",
    )
    draw_edge_interval_figure(edge_intervals, output_dir / "edge_frequency_wilson_intervals.png")
    draw_vni_distribution(vni, output_dir / "vni_net_strength_distribution.png")
    return {
        "graphs": graphs,
        "edge_intervals": edge_intervals,
        "role_intervals": role_intervals,
        "pairwise": pairwise,
        "consensus": consensus,
        "vni": vni,
    }
