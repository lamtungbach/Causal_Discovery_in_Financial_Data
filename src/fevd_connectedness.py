from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from config import CAMERA_DATA_DIR, CAMERA_LAGS, CAMERA_TABLE_DIR, LABELS, RETURN_COLS
from src.camera_ready_data import prepare_supervised


DISPLAY_LABELS = [LABELS[column] for column in RETURN_COLS]
HORIZONS = (10, 20)


def moving_average_matrices(coefs: np.ndarray, horizon: int) -> np.ndarray:
    """Return Phi_0,...,Phi_(H-1) for a fitted VAR coefficient array."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    p, n, n_again = coefs.shape
    if n != n_again:
        raise ValueError("VAR coefficient matrices must be square")
    phi = np.zeros((horizon, n, n), dtype=float)
    phi[0] = np.eye(n)
    for step in range(1, horizon):
        for lag in range(1, min(p, step) + 1):
            phi[step] += coefs[lag - 1] @ phi[step - lag]
    return phi


def generalized_fevd(coefs: np.ndarray, sigma: np.ndarray, horizon: int) -> np.ndarray:
    """Compute row-normalized Pesaran-Shin generalized FEVD.

    Entry (i,j) is the share of the H-step forecast-error variance of receiver
    i attributable to shock j. Generalized shocks are scaled by sigma[j,j]
    and are not orthogonalized, so the result does not depend on variable order.
    """
    sigma = np.asarray(sigma, dtype=float)
    phi = moving_average_matrices(np.asarray(coefs, dtype=float), horizon)
    n = sigma.shape[0]
    if sigma.shape != (n, n) or coefs.shape[1:] != (n, n):
        raise ValueError("coefs and sigma dimensions do not agree")
    diagonal = np.diag(sigma)
    if np.any(diagonal <= 0):
        raise ValueError("innovation covariance diagonal must be positive")

    raw = np.zeros((n, n), dtype=float)
    denominators = np.zeros(n, dtype=float)
    for impulse in phi:
        response_covariance = impulse @ sigma
        raw += response_covariance**2 / diagonal[np.newaxis, :]
        denominators += np.diag(response_covariance @ impulse.T)
    if np.any(denominators <= 0):
        raise RuntimeError("non-positive forecast-error variance denominator")
    raw /= denominators[:, np.newaxis]
    raw = np.maximum(raw, 0.0)
    row_sums = raw.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise RuntimeError("generalized FEVD row has zero mass")
    return raw / row_sums


def fit_gfevd(development: pd.DataFrame, lags: int = CAMERA_LAGS) -> tuple[object, dict[int, pd.DataFrame]]:
    numeric = development.astype(float)
    fitted = VAR(numeric).fit(maxlags=lags, ic=None, trend="c")
    if fitted.k_ar != lags:
        raise RuntimeError(f"Expected VAR({lags}), fitted VAR({fitted.k_ar})")
    labels = numeric.columns.tolist()
    matrices = {
        horizon: pd.DataFrame(
            generalized_fevd(fitted.coefs, np.asarray(fitted.sigma_u), horizon),
            index=labels,
            columns=labels,
        )
        for horizon in HORIZONS
    }
    return fitted, matrices


def labelled_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index.name = "receiver"
    return result.reset_index()


def connectedness_tables(matrices: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    vni_rows: list[dict] = []
    for horizon, frame in matrices.items():
        values = frame.to_numpy(dtype=float)
        labels = frame.index.tolist()
        n = len(labels)
        from_values = 100.0 * (values.sum(axis=1) - np.diag(values))
        to_values = 100.0 * (values.sum(axis=0) - np.diag(values))
        net_values = to_values - from_values
        tci = float(from_values.sum() / n)
        ranks = pd.Series(net_values, index=labels).rank(ascending=False, method="min").astype(int)
        summary_rows.append(
            {
                "horizon": horizon,
                "record_type": "system",
                "node": "ALL",
                "total_connectedness_index": tci,
            }
        )
        for idx, node in enumerate(labels):
            summary_rows.append(
                {
                    "horizon": horizon,
                    "record_type": "node",
                    "node": node,
                    "from_connectedness": from_values[idx],
                    "to_connectedness": to_values[idx],
                    "net_connectedness": net_values[idx],
                    "net_rank": int(ranks[node]),
                    "role": "Transmitter" if net_values[idx] > 1e-12 else "Receiver" if net_values[idx] < -1e-12 else "Balanced",
                }
            )
        for source_idx, source in enumerate(labels):
            for target_idx in range(source_idx + 1, n):
                target = labels[target_idx]
                # Positive means source is a net transmitter to target.
                pairwise_net = 100.0 * (values[target_idx, source_idx] - values[source_idx, target_idx])
                summary_rows.append(
                    {
                        "horizon": horizon,
                        "record_type": "pairwise",
                        "source": source,
                        "target": target,
                        "pairwise_net_connectedness": pairwise_net,
                    }
                )

        vni_idx = labels.index("VNI")
        vni_from = from_values[vni_idx]
        incoming = pd.Series(100.0 * values[vni_idx], index=labels).drop("VNI")
        incoming_rank = incoming.rank(ascending=False, method="min").astype(int)
        for source, contribution in incoming.items():
            vni_rows.append(
                {
                    "horizon": horizon,
                    "source": source,
                    "incoming_contribution": contribution,
                    "share_of_vni_from_connectedness": contribution / vni_from if vni_from > 0 else np.nan,
                    "incoming_rank": int(incoming_rank[source]),
                    "vni_from_connectedness": vni_from,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(vni_rows)


def _signed(value: float, tolerance: float = 1e-12) -> str:
    return "Transmitter" if value > tolerance else "Receiver" if value < -tolerance else "Balanced"


def cross_method_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    ngc = pd.read_csv(CAMERA_TABLE_DIR / "node_roles_prediction_weights.csv").set_index("node")
    var = pd.read_csv(CAMERA_TABLE_DIR / "var20_fdr_adjacency.csv", index_col="source")
    var = var.loc[DISPLAY_LABELS, DISPLAY_LABELS].astype(float)
    var_net = var.sum(axis=1) - var.sum(axis=0)
    var_rank = var_net.rank(ascending=False, method="min").astype(int)
    ngc_rank = ngc["net_strength"].rank(ascending=False, method="min").astype(int)

    rows = []
    nodes = summary.loc[summary["record_type"].eq("node")]
    for row in nodes.itertuples(index=False):
        rows.append(
            {
                "horizon": row.horizon,
                "node": row.node,
                "fevd_net_connectedness": row.net_connectedness,
                "fevd_role": row.role,
                "fevd_net_rank": row.net_rank,
                "var_fdr_net_degree": float(var_net[row.node]),
                "var_fdr_role": _signed(float(var_net[row.node])),
                "var_fdr_net_rank": int(var_rank[row.node]),
                "ngc_net_strength": float(ngc.loc[row.node, "net_strength"]),
                "ngc_role": str(ngc.loc[row.node, "role"]),
                "ngc_net_rank": int(ngc_rank[row.node]),
                "fevd_var_role_agreement": row.role == _signed(float(var_net[row.node])),
                "fevd_ngc_role_agreement": row.role == str(ngc.loc[row.node, "role"]),
            }
        )
    return pd.DataFrame(rows)


def permutation_checks(development: pd.DataFrame, reference: dict[int, pd.DataFrame]) -> pd.DataFrame:
    permutations = (
        list(reversed(DISPLAY_LABELS)),
        DISPLAY_LABELS[2:] + DISPLAY_LABELS[:2],
    )
    rows = []
    for number, order in enumerate(permutations, start=1):
        _, permuted = fit_gfevd(development[order])
        for horizon in HORIZONS:
            restored = permuted[horizon].loc[DISPLAY_LABELS, DISPLAY_LABELS]
            maximum_difference = float(
                np.max(np.abs(restored.to_numpy() - reference[horizon].to_numpy()))
            )
            rows.append(
                {
                    "permutation": number,
                    "horizon": horizon,
                    "maximum_absolute_difference": maximum_difference,
                    "passed_tolerance_1e_10": maximum_difference <= 1e-10,
                }
            )
    return pd.DataFrame(rows)


def draw_heatmaps(matrices: dict[int, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    maximum = max(frame.to_numpy().max() for frame in matrices.values())
    image = None
    for ax, horizon in zip(axes, HORIZONS):
        frame = matrices[horizon]
        image = ax.imshow(frame.to_numpy(), cmap="Blues", vmin=0, vmax=maximum)
        ax.set_xticks(range(len(frame)), frame.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(frame)), frame.index)
        ax.set_title(f"Generalized FEVD, H={horizon}")
        ax.set_xlabel("Shock source")
        ax.set_ylabel("Forecast-error receiver")
    fig.colorbar(image, ax=axes, label="Row-normalized variance share", shrink=0.8)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_net_connectedness(summary: pd.DataFrame, path: Path) -> None:
    nodes = summary.loc[summary["record_type"].eq("node")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, constrained_layout=True)
    for ax, horizon in zip(axes, HORIZONS):
        frame = nodes.loc[nodes["horizon"].eq(horizon)].sort_values("net_connectedness")
        colors = np.where(frame["net_connectedness"] >= 0, "#E67E22", "#2E86C1")
        ax.barh(frame["node"], frame["net_connectedness"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"NET connectedness, H={horizon}")
        ax.set_xlabel("TO minus FROM (percentage points)")
        ax.grid(axis="x", alpha=0.25)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_report(
    fitted: object,
    summary: pd.DataFrame,
    vni: pd.DataFrame,
    comparison: pd.DataFrame,
    checks: pd.DataFrame,
    development_start: str,
    development_end: str,
) -> str:
    systems = summary.loc[summary["record_type"].eq("system"), ["horizon", "total_connectedness_index"]]
    node_rows = summary.loc[summary["record_type"].eq("node")]
    compact_nodes = node_rows[["horizon", "node", "from_connectedness", "to_connectedness", "net_connectedness", "net_rank", "role"]]
    return "\n".join(
        [
            "# Diebold-Yilmaz Generalized FEVD Benchmark",
            "",
            "## Fixed specification",
            "",
            f"- Development sample: {development_start} through {development_end}; VAR observations after lags: {int(fitted.nobs)}.",
            f"- VAR lag order: {int(fitted.k_ar)}, identical to the canonical VAR-FDR benchmark.",
            "- Forecast horizons were pre-specified at H=10 and H=20. No rolling estimation was performed.",
            "",
            "## Formula and orientation",
            "",
            r"For receiver i and shock j, the generalized (Pesaran-Shin) H-step share before normalization is theta_ij(H) = sigma_jj^(-1) sum_{h=0}^{H-1}(e_i' Phi_h Sigma e_j)^2 / sum_{h=0}^{H-1}(e_i' Phi_h Sigma Phi_h' e_i). Each row is then divided by its row sum. Rows are receivers and columns are shock sources.",
            "",
            "This generalized decomposition does not use a Cholesky factor. The two explicit refits below verify ordering invariance after labels are restored.",
            "",
            "## Total connectedness index",
            "",
            systems.to_markdown(index=False),
            "",
            "## Directional connectedness",
            "",
            compact_nodes.to_markdown(index=False),
            "",
            "FROM is the off-diagonal share received by a node; TO is the column contribution sent to other nodes; NET = TO - FROM. Positive NET denotes a transmitter role.",
            "",
            "## VNI incoming connectedness",
            "",
            vni.to_markdown(index=False),
            "",
            "## Ordering-invariance checks",
            "",
            checks.to_markdown(index=False),
            "",
            "## Cross-method interpretation",
            "",
            comparison.to_markdown(index=False),
            "",
            "Only node-role signs and within-method ranks are compared. FEVD variance shares are not numerically compared with NGC ablation importance or binary VAR-FDR degree.",
            "",
            "## Interpretation boundary",
            "",
            "- The modeled variables are daily returns. These results measure return forecast-error connectedness, not volatility spillovers.",
            "- Generalized FEVD is an econometric connectedness benchmark; it does not replace or reinterpret the frozen NGC or VAR-FDR results.",
            "- The exercise is static over the fixed development sample and does not support time-varying or causal claims.",
        ]
    )


def run_fevd(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    returns = pd.read_csv(CAMERA_DATA_DIR / "main_return_panel.csv", parse_dates=["date"])
    split = prepare_supervised(returns, CAMERA_LAGS)
    development_raw = returns.iloc[: split.val_end][["date", *RETURN_COLS]].copy()
    development = development_raw[RETURN_COLS].rename(columns=LABELS).astype(float)
    fitted, matrices = fit_gfevd(development)
    for horizon, frame in matrices.items():
        labelled_matrix(frame).to_csv(output_dir / f"gfevd_h{horizon}.csv", index=False)

    summary, vni = connectedness_tables(matrices)
    comparison = cross_method_comparison(summary)
    checks = permutation_checks(development, matrices)
    if not checks["passed_tolerance_1e_10"].all():
        raise RuntimeError("Generalized FEVD failed an ordering-invariance check")
    summary.to_csv(output_dir / "connectedness_summary.csv", index=False)
    vni.to_csv(output_dir / "vni_connectedness.csv", index=False)
    comparison.to_csv(output_dir / "cross_method_network_comparison.csv", index=False)
    draw_heatmaps(matrices, output_dir / "gfevd_heatmaps.png")
    draw_net_connectedness(summary, output_dir / "net_connectedness.png")
    report = render_report(
        fitted,
        summary,
        vni,
        comparison,
        checks,
        development_raw["date"].iloc[0].date().isoformat(),
        development_raw["date"].iloc[-1].date().isoformat(),
    )
    (output_dir / "fevd_method_and_results.md").write_text(report, encoding="utf-8")
    return {"matrices": matrices, "summary": summary, "vni": vni, "comparison": comparison, "checks": checks}
