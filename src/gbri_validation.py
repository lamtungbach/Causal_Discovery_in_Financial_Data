from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from config import (
    CAMERA_GBRI_ALPHA,
    CAMERA_LAGS,
    CAMERA_READY_DIR,
    CAMERA_VOL_WINDOW,
    RETURN_COLS,
    TARGET_COL,
)
from src.camera_ready_analysis import DISPLAY_LABELS, compute_gbri
from src.camera_ready_data import prepare_supervised


TABLE_DIR = CAMERA_READY_DIR / "tables"
DATA_DIR = CAMERA_READY_DIR / "data"
HORIZONS = (5, 10)
EVENT_QUANTILE = 0.85
BOOTSTRAP_REPLICATIONS = 1000
BOOTSTRAP_BLOCK_LENGTH = 10
BOOTSTRAP_SEED = 20261025
SCORE_COLUMNS = {
    "GBRI": "gbri",
    "VNI volatility 20d": "vni_volatility_20d",
    "VIX absolute return": "vix_absolute_return",
}


def load_and_validate_canonical_gbri() -> tuple[pd.DataFrame, pd.DataFrame, object]:
    returns = pd.read_csv(DATA_DIR / "main_return_panel.csv", parse_dates=["date"])
    saved = pd.read_csv(TABLE_DIR / "gbri_prediction_weights.csv", parse_dates=["date"])
    retained_frame = pd.read_csv(TABLE_DIR / "ablation_importance_retained_seed42.csv")
    retained = retained_frame.set_index("source").loc[DISPLAY_LABELS, DISPLAY_LABELS].to_numpy(float)
    split = prepare_supervised(returns, CAMERA_LAGS)
    reconstructed, _, _ = compute_gbri(
        returns,
        retained,
        split.train_end,
        split.val_end,
        split.mean,
        split.std,
        alpha=CAMERA_GBRI_ALPHA,
        volatility_window=CAMERA_VOL_WINDOW,
    )
    if not reconstructed["date"].equals(saved["date"]):
        raise AssertionError("Reconstructed GBRI dates differ from canonical saved dates")
    numeric_columns = [
        "incoming_prediction_pressure",
        "vni_volatility_20d",
        "pressure_z_train_reference",
        "volatility_z_train_reference",
        "combined_risk_statistic",
        "gbri",
    ]
    np.testing.assert_allclose(
        reconstructed[numeric_columns].to_numpy(float),
        saved[numeric_columns].to_numpy(float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    test_mask = saved["segment"].eq("test")
    np.testing.assert_allclose(
        reconstructed.loc[test_mask, "gbri"],
        saved.loc[test_mask, "gbri"],
        rtol=0.0,
        atol=1e-12,
    )
    return returns, saved, split


def _forward_rms(values: np.ndarray, horizon: int) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    for index in range(0, len(values) - horizon):
        future = values[index + 1 : index + horizon + 1]
        output[index] = float(np.sqrt(np.mean(np.square(future))))
    return output


def _backward_rms(values: np.ndarray, horizon: int) -> np.ndarray:
    return (
        pd.Series(np.square(values))
        .rolling(horizon, min_periods=horizon)
        .mean()
        .pow(0.5)
        .to_numpy()
    )


def construct_forward_targets(
    returns: pd.DataFrame,
    canonical_gbri: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, float]]:
    if not returns["date"].equals(canonical_gbri["date"]):
        raise AssertionError("Return and GBRI dates are not aligned")
    frame = canonical_gbri.copy()
    frame[TARGET_COL] = returns[TARGET_COL].to_numpy(float)
    vix_column = RETURN_COLS[DISPLAY_LABELS.index("VIX")]
    frame["vix_absolute_return"] = returns[vix_column].abs().to_numpy(float)
    values = returns[TARGET_COL].to_numpy(float)
    thresholds: dict[int, float] = {}
    dates = pd.DatetimeIndex(frame["date"])

    for horizon in HORIZONS:
        target_col = f"forward_rv_{horizon}d"
        persistence_col = f"persistence_rv_{horizon}d"
        start_col = f"forward_{horizon}d_start_date"
        end_col = f"forward_{horizon}d_end_date"
        event_col = f"high_vol_event_{horizon}d"
        frame[target_col] = _forward_rms(values, horizon)
        frame[persistence_col] = _backward_rms(values, horizon)
        frame[start_col] = pd.NaT
        frame[end_col] = pd.NaT
        valid_indices = np.arange(0, len(frame) - horizon)
        frame.loc[valid_indices, start_col] = dates[valid_indices + 1].to_numpy()
        frame.loc[valid_indices, end_col] = dates[valid_indices + horizon].to_numpy()

        training_target = frame.loc[frame["segment"].eq("train"), target_col].dropna()
        threshold = float(training_target.quantile(EVENT_QUANTILE))
        thresholds[horizon] = threshold
        frame[f"training_event_threshold_{horizon}d"] = threshold
        frame[event_col] = np.where(
            frame[target_col].notna(),
            (frame[target_col] >= threshold).astype(float),
            np.nan,
        )
    return frame, thresholds


def _fit_training_calibrator(
    train_score: np.ndarray,
    train_event: np.ndarray,
) -> tuple[float, float, LogisticRegression | None, float]:
    mean = float(np.mean(train_score))
    std = float(np.std(train_score, ddof=0))
    base_rate = float(np.mean(train_event))
    if std <= 1e-12 or len(np.unique(train_event)) < 2:
        return mean, 1.0, None, base_rate
    model = LogisticRegression(solver="lbfgs", random_state=0)
    model.fit(((train_score - mean) / std).reshape(-1, 1), train_event.astype(int))
    return mean, std, model, base_rate


def _calibrated_probabilities(
    train_score: np.ndarray,
    train_event: np.ndarray,
    score: np.ndarray,
) -> np.ndarray:
    mean, std, model, base_rate = _fit_training_calibrator(train_score, train_event)
    if model is None:
        return np.full(len(score), base_rate, dtype=float)
    return model.predict_proba(((score - mean) / std).reshape(-1, 1))[:, 1]


def _metric_values(
    outcome: np.ndarray,
    event: np.ndarray,
    score: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    correlation = stats.spearmanr(score, outcome).statistic
    return {
        "spearman": float(correlation),
        "roc_auc": float(roc_auc_score(event, score)),
        "pr_auc": float(average_precision_score(event, score)),
        "brier": float(brier_score_loss(event, probability)),
    }


def moving_block_bootstrap_intervals(
    outcome: np.ndarray,
    event: np.ndarray,
    score: np.ndarray,
    probability: np.ndarray,
    *,
    replications: int = BOOTSTRAP_REPLICATIONS,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, tuple[float, float]]:
    n = len(outcome)
    if n < block_length:
        raise ValueError("Evaluation sample shorter than bootstrap block")
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("spearman", "roc_auc", "pr_auc", "brier")}
    blocks_needed = int(np.ceil(n / block_length))
    for _ in range(replications):
        starts = rng.integers(0, n - block_length + 1, size=blocks_needed)
        indices = np.concatenate(
            [np.arange(start, start + block_length) for start in starts]
        )[:n]
        sampled_event = event[indices]
        if len(np.unique(sampled_event)) < 2:
            continue
        metrics = _metric_values(
            outcome[indices], sampled_event, score[indices], probability[indices]
        )
        for name, value in metrics.items():
            if np.isfinite(value):
                values[name].append(value)
    intervals: dict[str, tuple[float, float]] = {}
    for name, samples in values.items():
        if len(samples) < max(100, replications // 2):
            raise RuntimeError(f"Insufficient valid bootstrap replicates for {name}")
        intervals[name] = (
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        )
    return intervals


def evaluate_models(
    targets: pd.DataFrame,
    thresholds: dict[int, float],
) -> pd.DataFrame:
    rows: list[dict] = []
    for horizon in HORIZONS:
        outcome_col = f"forward_rv_{horizon}d"
        event_col = f"high_vol_event_{horizon}d"
        model_scores = {**SCORE_COLUMNS, "VNI RV persistence": f"persistence_rv_{horizon}d"}
        for model_name, score_col in model_scores.items():
            train = targets.loc[
                targets["segment"].eq("train")
                & targets[outcome_col].notna()
                & targets[event_col].notna()
                & targets[score_col].notna()
            ]
            test = targets.loc[
                targets["segment"].eq("test")
                & targets[outcome_col].notna()
                & targets[event_col].notna()
                & targets[score_col].notna()
            ].copy()
            train_score = train[score_col].to_numpy(float)
            train_event = train[event_col].to_numpy(int)
            score = test[score_col].to_numpy(float)
            event = test[event_col].to_numpy(int)
            outcome = test[outcome_col].to_numpy(float)
            probability = _calibrated_probabilities(train_score, train_event, score)
            metrics = _metric_values(outcome, event, score, probability)
            intervals = moving_block_bootstrap_intervals(
                outcome,
                event,
                score,
                probability,
                seed=BOOTSTRAP_SEED + horizon * 100 + list(model_scores).index(model_name),
            )
            rows.append(
                {
                    "horizon_days": horizon,
                    "model": model_name,
                    "score_column": score_col,
                    "train_calibration_n": len(train),
                    "test_n": len(test),
                    "training_event_threshold": thresholds[horizon],
                    "test_event_rate": float(event.mean()),
                    "spearman": metrics["spearman"],
                    "spearman_ci_lower": intervals["spearman"][0],
                    "spearman_ci_upper": intervals["spearman"][1],
                    "roc_auc": metrics["roc_auc"],
                    "roc_auc_ci_lower": intervals["roc_auc"][0],
                    "roc_auc_ci_upper": intervals["roc_auc"][1],
                    "pr_auc": metrics["pr_auc"],
                    "pr_auc_ci_lower": intervals["pr_auc"][0],
                    "pr_auc_ci_upper": intervals["pr_auc"][1],
                    "brier": metrics["brier"],
                    "brier_ci_lower": intervals["brier"][0],
                    "brier_ci_upper": intervals["brier"][1],
                    "ci_method": f"moving_block_bootstrap_{BOOTSTRAP_REPLICATIONS}_rep_block_{BOOTSTRAP_BLOCK_LENGTH}",
                    "probability_calibration": "training_only_univariate_logistic",
                }
            )
    return pd.DataFrame(rows)


def regime_outcomes(targets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    test = targets.loc[targets["segment"].eq("test")]
    for horizon in HORIZONS:
        outcome = f"forward_rv_{horizon}d"
        event = f"high_vol_event_{horizon}d"
        for regime in ("Low", "Medium", "High", "Extreme"):
            frame = test.loc[test["risk_regime"].eq(regime) & test[outcome].notna()]
            rows.append(
                {
                    "horizon_days": horizon,
                    "risk_regime": regime,
                    "n": len(frame),
                    "mean_forward_rv": float(frame[outcome].mean()),
                    "median_forward_rv": float(frame[outcome].median()),
                    "p75_forward_rv": float(frame[outcome].quantile(0.75)),
                    "high_vol_event_rate": float(frame[event].mean()),
                }
            )
    return pd.DataFrame(rows)


def draw_metric_figure(metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    metric_specs = (
        ("spearman", "Spearman", True),
        ("roc_auc", "ROC-AUC", True),
        ("pr_auc", "PR-AUC", True),
        ("brier", "Brier score", False),
    )
    models = metrics["model"].drop_duplicates().tolist()
    x = np.arange(len(models))
    width = 0.36
    for ax, (column, title, higher_better) in zip(axes, metric_specs):
        for offset, horizon in ((-width / 2, 5), (width / 2, 10)):
            frame = metrics.loc[metrics["horizon_days"].eq(horizon)].set_index("model").loc[models]
            ax.bar(x + offset, frame[column], width, label=f"h={horizon}")
        ax.set_xticks(x, models, rotation=40, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if column in {"roc_auc", "pr_auc"}:
            ax.set_ylim(0, 1)
        if not higher_better:
            ax.set_ylabel("Lower is better")
    axes[0].legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_regime_figure(regimes: pd.DataFrame, path: Path) -> None:
    order = ["Low", "Medium", "High", "Extreme"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, horizon in zip(axes, HORIZONS):
        frame = regimes.loc[regimes["horizon_days"].eq(horizon)].set_index("risk_regime").loc[order]
        ax.bar(order, frame["mean_forward_rv"], color=["#6BAED6", "#74C476", "#FDAE6B", "#E6550D"])
        ax.set_title(f"Forward {horizon}-day realized volatility")
        ax.set_ylabel("Mean RMS return (%)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_report(
    metrics: pd.DataFrame,
    regimes: pd.DataFrame,
    thresholds: dict[int, float],
) -> str:
    comparisons: list[dict] = []
    for horizon in HORIZONS:
        frame = metrics.loc[metrics["horizon_days"].eq(horizon)].set_index("model")
        gbri = frame.loc["GBRI"]
        baselines = frame.drop(index="GBRI")
        comparisons.append(
            {
                "horizon_days": horizon,
                "gbri_best_spearman": bool(gbri["spearman"] >= baselines["spearman"].max()),
                "gbri_best_roc_auc": bool(gbri["roc_auc"] >= baselines["roc_auc"].max()),
                "gbri_best_pr_auc": bool(gbri["pr_auc"] >= baselines["pr_auc"].max()),
                "gbri_best_brier": bool(gbri["brier"] <= baselines["brier"].min()),
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    return "\n".join(
        [
            "# GBRI Forward-Risk Validation",
            "",
            "## Design",
            "",
            "- Evaluation uses the frozen seed-42 main graph and the canonical training-referenced GBRI.",
            "- Forward realized volatility is the root mean square of strictly later VNI returns at t+1 through t+h for pre-specified h=5 and h=10.",
            f"- A high-volatility event is defined separately by horizon using the training-only q{int(EVENT_QUANTILE * 100)} forward-volatility threshold: h5={thresholds[5]:.6f}, h10={thresholds[10]:.6f}.",
            "- Brier probabilities use a univariate logistic calibrator fitted only on training scores and training events.",
            f"- 95% evaluation intervals use {BOOTSTRAP_REPLICATIONS} moving-block bootstrap resamples of the fixed test predictions with block length {BOOTSTRAP_BLOCK_LENGTH}; models and calibrators are not refit in bootstrap resamples.",
            "",
            "## Test metrics",
            "",
            metrics.to_markdown(index=False),
            "",
            "## Does GBRI beat every simple baseline?",
            "",
            comparison_frame.to_markdown(index=False),
            "",
            "## Test outcomes by canonical GBRI regime",
            "",
            regimes.to_markdown(index=False),
            "",
            "## Interpretation boundary",
            "",
            "- Metrics describe predictive association on the frozen test period, not causality.",
            "- No horizon, event quantile, graph, alpha, regime threshold or endpoint was selected after seeing these results.",
            "- If GBRI does not consistently outperform the simple baselines, it should remain an exploratory monitoring index rather than a validated standalone risk predictor.",
            "- The final h observations lack a complete future window and are excluded for that horizon.",
        ]
    )


def run_validation(output_dir: Path) -> dict[str, object]:
    returns, canonical_gbri, split = load_and_validate_canonical_gbri()
    targets, thresholds = construct_forward_targets(returns, canonical_gbri)
    metrics = evaluate_models(targets, thresholds)
    regimes = regime_outcomes(targets)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_dir / "forward_volatility_targets.csv", index=False)
    metrics.to_csv(output_dir / "gbri_baseline_metrics.csv", index=False)
    regimes.to_csv(output_dir / "gbri_regime_outcomes.csv", index=False)
    (output_dir / "gbri_validation.md").write_text(
        render_report(metrics, regimes, thresholds), encoding="utf-8"
    )
    draw_metric_figure(metrics, output_dir / "gbri_baseline_metrics.png")
    draw_regime_figure(regimes, output_dir / "gbri_regime_outcomes.png")
    return {
        "returns": returns,
        "canonical_gbri": canonical_gbri,
        "split": split,
        "targets": targets,
        "thresholds": thresholds,
        "metrics": metrics,
        "regimes": regimes,
    }
