from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR

from config import LABELS, RETURN_COLS, TARGET_COL
from src.metrics import regression_metrics


DISPLAY_LABELS = [LABELS[column] for column in RETURN_COLS]


def matrix_frame(matrix: np.ndarray, labels: list[str] = DISPLAY_LABELS) -> pd.DataFrame:
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    return frame.reset_index().rename(columns={"index": "source"})


def fit_var_granger(
    returns: pd.DataFrame,
    development_end: int,
    lags: int,
) -> tuple[pd.DataFrame, np.ndarray, object]:
    """Fit one multivariate VAR(p) and run joint source-to-target F tests."""
    development = returns.iloc[:development_end][RETURN_COLS].astype(float)
    fitted = VAR(development).fit(maxlags=lags, ic=None, trend="c")
    if fitted.k_ar != lags:
        raise RuntimeError(f"Expected VAR({lags}), fitted VAR({fitted.k_ar})")

    rows = []
    for source_index, source in enumerate(RETURN_COLS):
        for target_index, target in enumerate(RETURN_COLS):
            if source_index == target_index:
                continue
            test = fitted.test_causality(caused=target, causing=source, kind="f")
            rows.append(
                {
                    "source": LABELS[source],
                    "target": LABELS[target],
                    "f_statistic": float(test.test_statistic),
                    "p_value": float(test.pvalue),
                    "var_lags": fitted.k_ar,
                    "development_observations": int(fitted.nobs),
                }
            )
    table = pd.DataFrame(rows)
    table["p_fdr_bh"] = multipletests(table["p_value"], method="fdr_bh")[1]
    table["edge_p05"] = table["p_value"] < 0.05
    table["edge_fdr05"] = table["p_fdr_bh"] < 0.05
    adjacency = np.zeros((len(RETURN_COLS), len(RETURN_COLS)), dtype=float)
    for row in table.itertuples(index=False):
        source = DISPLAY_LABELS.index(row.source)
        target = DISPLAY_LABELS.index(row.target)
        adjacency[source, target] = float(row.edge_fdr05)
    return table, adjacency, fitted


def graph_node_metrics(
    retained: np.ndarray,
    labels: list[str] = DISPLAY_LABELS,
) -> tuple[pd.DataFrame, nx.DiGraph]:
    graph = nx.DiGraph()
    graph.add_nodes_from(labels)
    for source, source_label in enumerate(labels):
        for target, target_label in enumerate(labels):
            weight = float(retained[source, target])
            if source != target and weight > 0:
                graph.add_edge(
                    source_label,
                    target_label,
                    weight=weight,
                    distance=1.0 / max(weight, 1e-12),
                )

    pagerank = nx.pagerank(graph, weight="weight", max_iter=1000)
    betweenness = nx.betweenness_centrality(graph, weight="distance")
    rows = []
    for node in labels:
        incoming = sum(data["weight"] for _, _, data in graph.in_edges(node, data=True))
        outgoing = sum(data["weight"] for _, _, data in graph.out_edges(node, data=True))
        net = outgoing - incoming
        role = "Transmitter" if net > 1e-12 else "Receiver" if net < -1e-12 else "Balanced"
        rows.append(
            {
                "node": node,
                "in_degree": graph.in_degree(node),
                "out_degree": graph.out_degree(node),
                "in_strength": incoming,
                "out_strength": outgoing,
                "net_strength": net,
                "pagerank": pagerank[node],
                "betweenness": betweenness[node],
                "role": role,
            }
        )
    return pd.DataFrame(rows).sort_values("net_strength", ascending=False), graph


def draw_prediction_graph(graph: nx.DiGraph, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    position = nx.spring_layout(graph, seed=42, weight="weight")
    node_colors = [
        "#E9A23B" if graph.out_degree(node) > graph.in_degree(node) else "#4CB7A5"
        for node in graph.nodes
    ]
    weights = [data["weight"] for _, _, data in graph.edges(data=True)]
    maximum = max(weights) if weights else 1.0
    nx.draw_networkx_nodes(graph, position, node_color=node_colors, node_size=1250, ax=ax)
    nx.draw_networkx_labels(graph, position, font_size=9, ax=ax)
    nx.draw_networkx_edges(
        graph,
        position,
        width=[1.0 + 3.0 * weight / maximum for weight in weights],
        edge_color=weights if weights else "#777777",
        edge_cmap=plt.cm.viridis,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        ax=ax,
    )
    edge_labels = {
        (source, target): f"{data['weight']:.3f}"
        for source, target, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(graph, position, edge_labels=edge_labels, font_size=7, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compare_binary_graphs(ngc: np.ndarray, var: np.ndarray) -> dict:
    mask = ~np.eye(ngc.shape[0], dtype=bool)
    ngc_edges = set(map(tuple, np.argwhere((ngc > 0) & mask)))
    var_edges = set(map(tuple, np.argwhere((var > 0) & mask)))
    union = ngc_edges | var_edges
    intersection = ngc_edges & var_edges
    vni = DISPLAY_LABELS.index("VNI")
    return {
        "ngc_edges": len(ngc_edges),
        "var_fdr_edges": len(var_edges),
        "edge_intersection": len(intersection),
        "edge_union": len(union),
        "jaccard": len(intersection) / len(union) if union else 1.0,
        "ngc_incoming_vni": [DISPLAY_LABELS[s] for s, t in sorted(ngc_edges) if t == vni],
        "var_incoming_vni": [DISPLAY_LABELS[s] for s, t in sorted(var_edges) if t == vni],
    }


def modified_diebold_mariano(
    y_true: np.ndarray,
    forecast_ngc: np.ndarray,
    forecast_lstm: np.ndarray,
    horizon: int = 1,
) -> dict:
    """Two-sided Harvey-Leybourne-Newbold modified DM test.

    The loss differential is NGC squared error minus LSTM squared error. For
    one-step non-overlapping forecasts the long-run variance uses lag zero.
    """
    actual = np.asarray(y_true, dtype=float)
    ngc = np.asarray(forecast_ngc, dtype=float)
    lstm = np.asarray(forecast_lstm, dtype=float)
    differential = (actual - ngc) ** 2 - (actual - lstm) ** 2
    n = len(differential)
    if n < 3:
        return {"mean_loss_diff": float("nan"), "statistic": float("nan"), "p_value": float("nan")}
    centered = differential - differential.mean()
    max_lag = max(0, horizon - 1)
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, max_lag + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_variance += 2.0 * covariance
    if long_run_variance <= 0:
        statistic = float("nan")
        p_value = float("nan")
    else:
        dm = differential.mean() / np.sqrt(long_run_variance / n)
        correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
        statistic = float(dm * correction)
        p_value = float(2.0 * stats.t.sf(abs(statistic), df=n - 1))
    return {
        "mean_loss_diff": float(differential.mean()),
        "statistic": statistic,
        "p_value": p_value,
        "horizon": horizon,
        "loss": "squared_error",
        "alternative": "two-sided",
        "n": n,
    }


def exact_mcnemar(
    y_true: np.ndarray,
    forecast_ngc: np.ndarray,
    forecast_lstm: np.ndarray,
) -> dict:
    correct_ngc = np.sign(y_true) == np.sign(forecast_ngc)
    correct_lstm = np.sign(y_true) == np.sign(forecast_lstm)
    b = int(np.sum(correct_lstm & ~correct_ngc))
    c = int(np.sum(correct_ngc & ~correct_lstm))
    discordant = b + c
    p_value = float(stats.binomtest(min(b, c), discordant, 0.5).pvalue) if discordant else 1.0
    return {"b_lstm_correct_ngc_wrong": b, "c_ngc_correct_lstm_wrong": c, "p_value": p_value, "n": len(y_true)}


def pesaran_timmermann(y_true: np.ndarray, forecast: np.ndarray) -> dict:
    """Binary up/non-up Pesaran-Timmermann directional test."""
    actual_up = np.asarray(y_true) > 0
    forecast_up = np.asarray(forecast) > 0
    n = len(actual_up)
    p_actual = float(actual_up.mean())
    p_forecast = float(forecast_up.mean())
    p_correct = float((actual_up == forecast_up).mean())
    p_independent = p_actual * p_forecast + (1 - p_actual) * (1 - p_forecast)
    variance_correct = p_independent * (1 - p_independent) / n
    variance_independent = (
        (2 * p_actual - 1) ** 2 * p_forecast * (1 - p_forecast) / n
        + (2 * p_forecast - 1) ** 2 * p_actual * (1 - p_actual) / n
        + 4 * p_actual * p_forecast * (1 - p_actual) * (1 - p_forecast) / (n**2)
    )
    denominator = variance_correct - variance_independent
    if denominator <= 0:
        statistic = float("nan")
        p_value = float("nan")
    else:
        statistic = float((p_correct - p_independent) / np.sqrt(denominator))
        p_value = float(2 * stats.norm.sf(abs(statistic)))
    return {
        "statistic": statistic,
        "p_value": p_value,
        "directional_accuracy": p_correct,
        "chance_accuracy": p_independent,
        "n": n,
        "zero_rule": "actual_or_forecast_zero_is_non_up",
    }


def forecast_results_and_inference(
    dates: np.ndarray,
    y_true_standardized: np.ndarray,
    lstm_standardized: np.ndarray,
    ngc_standardized: np.ndarray,
    target_mean: float,
    target_std: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    actual = y_true_standardized * target_std + target_mean
    lstm = lstm_standardized * target_std + target_mean
    ngc = ngc_standardized * target_std + target_mean
    predictions = pd.DataFrame(
        {"date": pd.to_datetime(dates), "y_true": actual, "lstm_pred": lstm, "ngc_pred": ngc}
    )
    metrics = pd.DataFrame(
        [
            {"model": "LSTM", **regression_metrics(actual, lstm)},
            {"model": "NGC", **regression_metrics(actual, ngc)},
        ]
    )
    inference = {
        "dm_squared_error_hln": modified_diebold_mariano(actual, ngc, lstm, horizon=1),
        "mcnemar_exact": exact_mcnemar(actual, ngc, lstm),
        "pesaran_timmermann": {
            "ngc": pesaran_timmermann(actual, ngc),
            "lstm": pesaran_timmermann(actual, lstm),
        },
    }
    return predictions, metrics, inference


def compute_gbri(
    returns: pd.DataFrame,
    retained: np.ndarray,
    train_end: int,
    val_end: int,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    alpha: float,
    volatility_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = returns[RETURN_COLS].to_numpy(dtype=float)
    standardized = (raw - mean) / std
    vni = RETURN_COLS.index(TARGET_COL)
    incoming_weights = retained[:, vni].copy()
    incoming_weights[vni] = 0.0
    pressure = np.abs(standardized) @ incoming_weights
    volatility = (
        pd.Series(raw[:, vni]).rolling(volatility_window).std(ddof=1).to_numpy()
    )
    indices = np.arange(len(raw))
    training_valid = (indices < train_end) & np.isfinite(pressure) & np.isfinite(volatility)
    if training_valid.sum() < 30:
        raise ValueError("Insufficient training observations for GBRI reference distribution")

    pressure_mean = float(np.mean(pressure[training_valid]))
    pressure_std = float(np.std(pressure[training_valid], ddof=0))
    volatility_mean = float(np.mean(volatility[training_valid]))
    volatility_std = float(np.std(volatility[training_valid], ddof=0))
    zero_pressure_variance = pressure_std <= 1e-12
    pressure_z = (
        np.zeros_like(pressure)
        if zero_pressure_variance
        else (pressure - pressure_mean) / pressure_std
    )
    if volatility_std <= 1e-12:
        raise ValueError("Training VNI volatility has zero variance")
    volatility_z = (volatility - volatility_mean) / volatility_std
    combined = alpha * pressure_z + (1 - alpha) * volatility_z
    reference = np.sort(combined[training_valid])
    gbri = 100.0 * np.searchsorted(reference, combined, side="right") / len(reference)
    gbri[~np.isfinite(combined)] = np.nan

    segment = np.where(indices < train_end, "train", np.where(indices < val_end, "validation", "test"))
    regime = pd.cut(
        gbri,
        bins=[-np.inf, 70, 85, 95, np.inf],
        labels=["Low", "Medium", "High", "Extreme"],
        right=False,
    )
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(returns["date"]),
            "segment": segment,
            "interpretation": np.where(segment == "test", "out_of_sample_monitoring", "retrospective_reconstruction"),
            "incoming_prediction_pressure": pressure,
            "vni_volatility_20d": volatility,
            "pressure_z_train_reference": pressure_z,
            "volatility_z_train_reference": volatility_z,
            "combined_risk_statistic": combined,
            "gbri": gbri,
            "risk_regime": regime,
        }
    )
    distributions = []
    for selected in ("all", "train", "validation", "test"):
        subset = output if selected == "all" else output.loc[output["segment"].eq(selected)]
        counts = subset["risk_regime"].value_counts().reindex(["Low", "Medium", "High", "Extreme"], fill_value=0)
        valid_n = int(subset["risk_regime"].notna().sum())
        for label, count in counts.items():
            distributions.append(
                {"segment": selected, "regime": label, "days": int(count), "share": float(count / valid_n) if valid_n else float("nan")}
            )
    metadata = {
        "alpha": alpha,
        "volatility_window": volatility_window,
        "training_reference_n": int(training_valid.sum()),
        "pressure_train_mean": pressure_mean,
        "pressure_train_std": pressure_std,
        "volatility_train_mean": volatility_mean,
        "volatility_train_std": volatility_std,
        "zero_pressure_variance": zero_pressure_variance,
        "incoming_vni_sources": [DISPLAY_LABELS[i] for i, weight in enumerate(incoming_weights) if weight > 0],
        "historical_scope_note": "Only the test segment is out-of-sample with respect to checkpoint and graph selection.",
    }
    return output, pd.DataFrame(distributions), metadata


def draw_gbri(gbri: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(gbri["date"], gbri["gbri"], linewidth=0.9, color="#2878B5")
    ax.axhline(70, color="#E59F00", linestyle="--", linewidth=0.8)
    ax.axhline(85, color="#D55E00", linestyle="--", linewidth=0.8)
    ax.axhline(95, color="#9B2226", linestyle="--", linewidth=0.8)
    test = gbri.loc[gbri["segment"].eq("test"), "date"]
    if len(test):
        ax.axvline(test.iloc[0], color="black", linestyle=":", linewidth=1.0, label="Test period begins")
        ax.legend(loc="upper left")
    ax.set_ylim(0, 101)
    ax.set_ylabel("GBRI percentile")
    ax.set_title("Graph-based VN-Index risk monitoring index")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_json(data: dict, path: Path) -> None:
    def sanitize(value):
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, np.integer):
            return int(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize(data), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
