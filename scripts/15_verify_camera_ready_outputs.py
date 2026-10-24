from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CAMERA_LAGS,
    CAMERA_LSTM_HIDDEN,
    CAMERA_NGC_HIDDEN,
    CAMERA_READY_DIR,
    CAMERA_STABILITY_THRESHOLD,
    RETURN_COLS,
    TARGET_COL,
)
from src.camera_ready_data import build_return_dataset, prepare_supervised  # noqa: E402
from src.camera_ready_models import TargetMLP, VniLSTM, predict_model  # noqa: E402


TABLES = CAMERA_READY_DIR / "tables"
MODELS = CAMERA_READY_DIR / "models"
FIGURES = CAMERA_READY_DIR / "figures"


def require(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"Missing or empty output: {path}")
    return path


def require_finite(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError(f"Non-finite values in {name}: {columns}")


def main() -> None:
    required = [
        CAMERA_READY_DIR / "data_audit.md",
        CAMERA_READY_DIR / "experiment_config.json",
        CAMERA_READY_DIR / "camera_ready_results_summary.md",
        TABLES / "main_split_summary.csv",
        TABLES / "invalid_return_endpoints.csv",
        TABLES / "test_predictions.csv",
        TABLES / "forecast_metrics.csv",
        TABLES / "forecast_inference.json",
        TABLES / "ablation_edge_importance_seed42.csv",
        TABLES / "ablation_importance_retained_seed42.csv",
        TABLES / "node_roles_prediction_weights.csv",
        TABLES / "var20_multivariate_granger.csv",
        TABLES / "edge_stability_20seeds.csv",
        TABLES / "node_role_stability_20seeds.csv",
        TABLES / "stability_run_summary.csv",
        TABLES / "calendar_timing_robustness.csv",
        TABLES / "gbri_prediction_weights.csv",
        TABLES / "gbri_regime_distribution.csv",
        TABLES / "reproducibility_check.json",
        MODELS / "ngc_seed42.pt",
        MODELS / "lstm_seed42.pt",
        FIGURES / "prediction_weighted_ngc_graph.png",
        FIGURES / "common_day_prediction_graph.png",
        FIGURES / "timing_shift_prediction_graph.png",
        FIGURES / "gbri_prediction_weights.png",
    ]
    for path in required:
        require(path)

    config = json.loads((CAMERA_READY_DIR / "experiment_config.json").read_text(encoding="utf-8"))
    if config["lag"] != 20 or config["ngc_hidden"] != [64, 64]:
        raise AssertionError("Experiment contract changed unexpectedly")
    if config["stability_seeds"] != list(range(42, 62)):
        raise AssertionError("Full camera-ready output must use the pre-specified 20 seeds")

    # If a user runs script 14 with a skip flag over an existing directory,
    # old stability/robustness files must not be mistaken for current outputs.
    config_time = (CAMERA_READY_DIR / "experiment_config.json").stat().st_mtime
    freshness_required = [
        CAMERA_READY_DIR / "camera_ready_results_summary.md",
        TABLES / "stability_run_summary.csv",
        TABLES / "calendar_timing_robustness.csv",
        MODELS / "ngc_seed42.pt",
        MODELS / "lstm_seed42.pt",
    ]
    stale = [str(path) for path in freshness_required if path.stat().st_mtime < config_time]
    if stale:
        raise AssertionError(f"Outputs older than the current experiment config: {stale}")

    split = pd.read_csv(TABLES / "main_split_summary.csv").iloc[0]
    expected = {
        "raw_observations": 2828,
        "train_forecast_origins": 1959,
        "validation_forecast_origins": 424,
        "test_forecast_origins": 425,
    }
    for key, value in expected.items():
        if int(split[key]) != value:
            raise AssertionError(f"Unexpected {key}: {split[key]} != {value}")
    if not (
        pd.Timestamp(split.train_end) < pd.Timestamp(split.validation_start)
        and pd.Timestamp(split.validation_end) < pd.Timestamp(split.test_start)
    ):
        raise AssertionError("Chronological split boundaries overlap")

    invalid = pd.read_csv(TABLES / "invalid_return_endpoints.csv", parse_dates=["date"])
    main_oil = invalid.loc[
        invalid["calendar"].eq("main") & invalid["asset"].eq("oil"), "date"
    ].dt.strftime("%Y-%m-%d").tolist()
    if main_oil != ["2020-04-20", "2020-04-21"]:
        raise AssertionError(f"Unexpected WTI invalid endpoints: {main_oil}")

    predictions = pd.read_csv(TABLES / "test_predictions.csv", parse_dates=["date"])
    if len(predictions) != 425 or predictions["date"].duplicated().any():
        raise AssertionError("Test predictions must contain 425 unique forecast dates")
    if not predictions["date"].is_monotonic_increasing:
        raise AssertionError("Test prediction dates are not chronological")
    require_finite(predictions, ["y_true", "lstm_pred", "ngc_pred"], "test predictions")

    metrics = pd.read_csv(TABLES / "forecast_metrics.csv")
    if set(metrics["model"]) != {"LSTM", "NGC"}:
        raise AssertionError("Forecast metrics must contain exactly LSTM and NGC")
    require_finite(metrics, ["RMSE", "MAE", "R2", "Directional_Accuracy"], "forecast metrics")

    inference = json.loads((TABLES / "forecast_inference.json").read_text(encoding="utf-8"))
    for branch in (
        inference["dm_squared_error_hln"],
        inference["mcnemar_exact"],
        inference["pesaran_timmermann"]["ngc"],
        inference["pesaran_timmermann"]["lstm"],
    ):
        p_value = branch.get("p_value")
        if p_value is not None and not 0 <= p_value <= 1:
            raise AssertionError(f"Invalid inference p-value: {p_value}")

    importance = pd.read_csv(TABLES / "ablation_edge_importance_seed42.csv")
    if len(importance) != 42 or importance[["source", "target"]].duplicated().any():
        raise AssertionError("Ablation audit must contain 42 unique off-diagonal edges")
    require_finite(
        importance,
        ["full_val_mse", "ablated_val_mse", "signed_relative_loss_change", "relative_loss_increase"],
        "ablation importance",
    )
    if (importance["relative_loss_increase"] < 0).any():
        raise AssertionError("Positive importance cannot be negative")

    retained = pd.read_csv(TABLES / "ablation_importance_retained_seed42.csv", index_col=0)
    matrix = retained.to_numpy(dtype=float)
    if matrix.shape != (7, 7) or not np.allclose(np.diag(matrix), 0):
        raise AssertionError("Retained adjacency must be a 7x7 source-row matrix with zero diagonal")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise AssertionError("Retained adjacency contains invalid weights")

    var = pd.read_csv(TABLES / "var20_multivariate_granger.csv")
    if len(var) != 42 or set(var["var_lags"]) != {20}:
        raise AssertionError("VAR output must contain 42 VAR(20) source-target tests")
    require_finite(var, ["f_statistic", "p_value", "p_fdr_bh"], "VAR tests")

    runs = pd.read_csv(TABLES / "stability_run_summary.csv")
    if len(runs) != 20 or runs["seed"].nunique() != 20:
        raise AssertionError("Stability output must contain 20 distinct seeds")
    expected_seed_files = list((TABLES / "stability_runs").glob("seed_*_importance.csv"))
    if len(expected_seed_files) != 20:
        raise AssertionError("Expected one importance audit for each of 20 seeds")
    stability = pd.read_csv(TABLES / "edge_stability_20seeds.csv")
    if len(stability) != 42:
        raise AssertionError("Edge stability must contain 42 off-diagonal edges")
    if not stability["selection_frequency"].between(0, 1).all():
        raise AssertionError("Selection frequencies must lie in [0, 1]")
    expected_stable = stability["selection_frequency"] >= CAMERA_STABILITY_THRESHOLD
    if not np.array_equal(expected_stable.to_numpy(), stability["stable"].to_numpy(dtype=bool)):
        raise AssertionError("Stable-edge flags do not match the pre-specified threshold")

    robustness = pd.read_csv(TABLES / "calendar_timing_robustness.csv")
    expected_variants = {
        "main_calendar",
        "common_trading_days_no_interpolation",
        "main_calendar_extra_global_input_shift",
    }
    if set(robustness["variant"]) != expected_variants:
        raise AssertionError("Calendar/timing robustness variants are incomplete")

    gbri = pd.read_csv(TABLES / "gbri_prediction_weights.csv")
    finite_gbri = gbri["gbri"].dropna()
    if not finite_gbri.between(0, 100).all():
        raise AssertionError("GBRI must lie in [0, 100]")
    if len(gbri.loc[gbri["segment"].eq("test")]) != 425:
        raise AssertionError("GBRI test segment must match 425 forecast dates")
    if not set(gbri["interpretation"].dropna()).issubset(
        {"out_of_sample_monitoring", "retrospective_reconstruction"}
    ):
        raise AssertionError("GBRI historical interpretation labels are invalid")

    ngc_checkpoint = torch.load(MODELS / "ngc_seed42.pt", map_location="cpu", weights_only=False)
    lstm_checkpoint = torch.load(MODELS / "lstm_seed42.pt", map_location="cpu", weights_only=False)
    if len(ngc_checkpoint["state_dicts"]) != 7:
        raise AssertionError("NGC checkpoint must contain seven target models")
    if lstm_checkpoint["lags"] != 20:
        raise AssertionError("LSTM checkpoint metadata is inconsistent")

    # Load the saved checkpoints into fresh model objects and reproduce the
    # exported predictions. This detects stale/incompatible model artifacts.
    _, returns, _, _ = build_return_dataset("main")
    reconstructed = prepare_supervised(returns, CAMERA_LAGS)
    target = RETURN_COLS.index(TARGET_COL)
    if not np.allclose(reconstructed.mean, ngc_checkpoint["mean"]):
        raise AssertionError("NGC checkpoint scaling metadata does not match the current data")
    if not np.allclose(reconstructed.std, ngc_checkpoint["std"]):
        raise AssertionError("NGC checkpoint scaling metadata does not match the current data")

    ngc_target = TargetMLP(len(RETURN_COLS), CAMERA_LAGS, CAMERA_NGC_HIDDEN)
    ngc_target.load_state_dict(ngc_checkpoint["state_dicts"][target])
    ngc_reloaded = (
        predict_model(ngc_target, reconstructed.X_test) * reconstructed.std[target]
        + reconstructed.mean[target]
    )
    lstm = VniLSTM(len(RETURN_COLS), CAMERA_LSTM_HIDDEN)
    lstm.load_state_dict(lstm_checkpoint["state_dict"])
    lstm_reloaded = (
        predict_model(lstm, reconstructed.X_test) * reconstructed.std[target]
        + reconstructed.mean[target]
    )
    expected_y = (
        reconstructed.Y_test[:, target] * reconstructed.std[target]
        + reconstructed.mean[target]
    )
    for label, actual, exported in (
        ("actual target", expected_y, predictions["y_true"].to_numpy()),
        ("NGC checkpoint", ngc_reloaded, predictions["ngc_pred"].to_numpy()),
        ("LSTM checkpoint", lstm_reloaded, predictions["lstm_pred"].to_numpy()),
    ):
        if not np.allclose(actual, exported, rtol=1e-6, atol=1e-7):
            difference = float(np.max(np.abs(actual - exported)))
            raise AssertionError(f"{label} does not reproduce exported predictions: {difference}")

    reproducibility = json.loads(
        (TABLES / "reproducibility_check.json").read_text(encoding="utf-8")
    )
    if reproducibility["max_abs_repeat_diff"] > 1e-10:
        raise AssertionError("Same-seed reproducibility check failed")
    if reproducibility["max_abs_different_seed_diff"] <= 1e-10:
        raise AssertionError("Different-seed check is degenerate")
    if reproducibility["seed42_prediction_std"] <= 1e-10:
        raise AssertionError("NGC predictions are constant")

    print("PASS: all camera-ready output integrity checks succeeded")
    print(f"Stable edges at frequency >= {CAMERA_STABILITY_THRESHOLD:.2f}: {int(stability['stable'].sum())}")
    print(f"Seed-42 retained edges: {int(np.count_nonzero(matrix))}")


if __name__ == "__main__":
    main()
