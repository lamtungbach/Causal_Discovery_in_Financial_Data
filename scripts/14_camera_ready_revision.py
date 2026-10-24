from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CAMERA_DATA_DIR,
    CAMERA_FIGURE_DIR,
    CAMERA_GBRI_ALPHA,
    CAMERA_GRAPH_QUANTILE,
    CAMERA_LAGS,
    CAMERA_LSTM_HIDDEN,
    CAMERA_MODEL_DIR,
    CAMERA_NGC_HIDDEN,
    CAMERA_NGC_LAMBDA,
    CAMERA_READY_DIR,
    CAMERA_STABILITY_SEEDS,
    CAMERA_STABILITY_THRESHOLD,
    CAMERA_TABLE_DIR,
    CAMERA_VOL_WINDOW,
    LABELS,
    RETURN_COLS,
    SEED,
    TARGET_COL,
)
from src.camera_ready_analysis import (  # noqa: E402
    DISPLAY_LABELS,
    compare_binary_graphs,
    compute_gbri,
    draw_gbri,
    draw_prediction_graph,
    fit_var_granger,
    forecast_results_and_inference,
    graph_node_metrics,
    matrix_frame,
    write_json,
)
from src.camera_ready_data import (  # noqa: E402
    SupervisedSplit,
    build_return_dataset,
    prepare_supervised,
    write_data_audit,
)
from src.camera_ready_models import (  # noqa: E402
    block_ablation_importance,
    predict_model,
    predict_ngc,
    resolve_device,
    sparsify_importance,
    train_lstm,
    train_ngc,
)
from src.metrics import regression_metrics  # noqa: E402


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path}")


def output_directories(smoke_test: bool) -> dict[str, Path]:
    if not smoke_test:
        return {
            "root": CAMERA_READY_DIR,
            "tables": CAMERA_TABLE_DIR,
            "figures": CAMERA_FIGURE_DIR,
            "models": CAMERA_MODEL_DIR,
            "data": CAMERA_DATA_DIR,
        }
    root = CAMERA_READY_DIR / "smoke"
    directories = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "models": root / "models",
        "data": root / "data",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def save_split_summary(split: SupervisedSplit, path: Path, variant: str) -> None:
    summary = {"variant": variant, **split.split_summary}
    save_csv(pd.DataFrame([summary]), path)


def train_variant_ngc(
    split: SupervisedSplit,
    *,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
    learning_rate: float,
) -> tuple[list, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, float, pd.DataFrame]:
    models, history, checkpoints = train_ngc(
        split.X_train,
        split.Y_train,
        split.X_val,
        split.Y_val,
        hidden=CAMERA_NGC_HIDDEN,
        lam=CAMERA_NGC_LAMBDA,
        seed=seed,
        device=device,
        epochs=epochs,
        patience=patience,
        lr=learning_rate,
    )
    importance, audit = block_ablation_importance(
        models, split.X_val, split.Y_val, DISPLAY_LABELS
    )
    retained, threshold = sparsify_importance(importance, CAMERA_GRAPH_QUANTILE)
    audit["retained"] = [
        retained[DISPLAY_LABELS.index(row.source), DISPLAY_LABELS.index(row.target)] > 0
        for row in audit.itertuples(index=False)
    ]
    return models, history, checkpoints, importance, retained, threshold, audit


def run_main_models(
    split: SupervisedSplit,
    returns: pd.DataFrame,
    directories: dict[str, Path],
    *,
    device: str,
    epochs: int,
    patience: int,
    learning_rate: float,
) -> dict:
    tables = directories["tables"]
    figures = directories["figures"]
    models_dir = directories["models"]
    target = RETURN_COLS.index(TARGET_COL)

    print("\n=== LSTM: main calendar ===")
    lstm_fit = train_lstm(
        split.X_train,
        split.Y_train[:, target],
        split.X_val,
        split.Y_val[:, target],
        hidden=CAMERA_LSTM_HIDDEN,
        seed=SEED,
        device=device,
        epochs=epochs,
        patience=patience,
        lr=learning_rate,
    )
    save_csv(lstm_fit.history, tables / "lstm_training_history.csv")

    print("\n=== NGC: main calendar ===")
    (
        ngc_models,
        ngc_history,
        ngc_checkpoints,
        importance,
        retained,
        threshold,
        importance_audit,
    ) = train_variant_ngc(
        split,
        seed=SEED,
        device=device,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
    )
    save_csv(ngc_history, tables / "ngc_training_history_seed42.csv")
    save_csv(ngc_checkpoints, tables / "ngc_checkpoints_seed42.csv")
    save_csv(importance_audit, tables / "ablation_edge_importance_seed42.csv")
    save_csv(matrix_frame(importance), tables / "ablation_importance_matrix_seed42.csv")
    save_csv(matrix_frame(retained), tables / "ablation_importance_retained_seed42.csv")

    node_metrics, graph = graph_node_metrics(retained)
    save_csv(node_metrics, tables / "node_roles_prediction_weights.csv")
    draw_prediction_graph(
        graph,
        figures / "prediction_weighted_ngc_graph.png",
        "Prediction-weighted Neural Granger network",
    )

    lstm_standardized = predict_model(lstm_fit.model, split.X_test)
    ngc_standardized_all = predict_ngc(ngc_models, split.X_test)
    predictions, forecast_metrics, inference = forecast_results_and_inference(
        split.dates_test,
        split.Y_test[:, target],
        lstm_standardized,
        ngc_standardized_all[:, target],
        split.mean[target],
        split.std[target],
    )
    save_csv(predictions, tables / "test_predictions.csv")
    save_csv(forecast_metrics, tables / "forecast_metrics.csv")
    write_json(inference, tables / "forecast_inference.json")

    all_target_metrics = []
    for target_index, label in enumerate(DISPLAY_LABELS):
        actual = split.Y_test[:, target_index] * split.std[target_index] + split.mean[target_index]
        prediction = ngc_standardized_all[:, target_index] * split.std[target_index] + split.mean[target_index]
        all_target_metrics.append({"target": label, **regression_metrics(actual, prediction)})
    save_csv(pd.DataFrame(all_target_metrics), tables / "ngc_all_target_forecast_metrics.csv")

    print("\n=== Multivariate VAR(20): main development sample ===")
    var_table, var_adjacency, fitted_var = fit_var_granger(returns, split.val_end, CAMERA_LAGS)
    save_csv(var_table, tables / "var20_multivariate_granger.csv")
    save_csv(matrix_frame(var_adjacency), tables / "var20_fdr_adjacency.csv")
    graph_comparison = compare_binary_graphs(retained, var_adjacency)
    write_json(graph_comparison, tables / "var_ngc_graph_comparison.json")

    gbri, gbri_distribution, gbri_metadata = compute_gbri(
        returns,
        retained,
        split.train_end,
        split.val_end,
        split.mean,
        split.std,
        alpha=CAMERA_GBRI_ALPHA,
        volatility_window=CAMERA_VOL_WINDOW,
    )
    save_csv(gbri, tables / "gbri_prediction_weights.csv")
    save_csv(gbri_distribution, tables / "gbri_regime_distribution.csv")
    write_json(gbri_metadata, tables / "gbri_metadata.json")
    draw_gbri(gbri, figures / "gbri_prediction_weights.png")

    checkpoint = {
        "model_type": "group_norm_regularized_target_mlp",
        "architecture": [CAMERA_LAGS * len(RETURN_COLS), *CAMERA_NGC_HIDDEN, 1],
        "lags": CAMERA_LAGS,
        "lambda": CAMERA_NGC_LAMBDA,
        "seed": SEED,
        "mean": split.mean,
        "std": split.std,
        "labels": DISPLAY_LABELS,
        "state_dicts": [model.state_dict() for model in ngc_models],
        "checkpoint_table": ngc_checkpoints.to_dict(orient="records"),
        "split_summary": split.split_summary,
    }
    torch.save(checkpoint, models_dir / "ngc_seed42.pt")
    torch.save(
        {
            "model_type": "lstm",
            "architecture": [len(RETURN_COLS), CAMERA_LSTM_HIDDEN, 1],
            "lags": CAMERA_LAGS,
            "seed": SEED,
            "mean": split.mean,
            "std": split.std,
            "state_dict": lstm_fit.model.state_dict(),
            "best_epoch": lstm_fit.best_epoch,
            "best_val_mse": lstm_fit.best_val_mse,
            "split_summary": split.split_summary,
        },
        models_dir / "lstm_seed42.pt",
    )

    return {
        "models": ngc_models,
        "importance": importance,
        "retained": retained,
        "threshold": threshold,
        "importance_audit": importance_audit,
        "node_metrics": node_metrics,
        "forecast_metrics": forecast_metrics,
        "inference": inference,
        "var_adjacency": var_adjacency,
        "var_lags": fitted_var.k_ar,
        "graph_comparison": graph_comparison,
        "gbri_distribution": gbri_distribution,
        "gbri_metadata": gbri_metadata,
    }


def run_stability(
    split: SupervisedSplit,
    directories: dict[str, Path],
    main_result: dict,
    *,
    seed_values: list[int],
    device: str,
    epochs: int,
    patience: int,
    learning_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"\n=== NGC stability: {len(seed_values)} seeds ===")
    run_dir = directories["tables"] / "stability_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    retained_runs = []
    importance_runs = []
    signed_runs = []
    role_runs = []
    run_summaries = []

    for run_index, seed in enumerate(seed_values, start=1):
        print(f"Stability seed {seed} ({run_index}/{len(seed_values)})")
        if seed == SEED:
            importance = main_result["importance"]
            retained = main_result["retained"]
            threshold = main_result["threshold"]
            audit = main_result["importance_audit"]
            nodes = main_result["node_metrics"]
        else:
            _, seed_history, seed_checkpoints, importance, retained, threshold, audit = train_variant_ngc(
                split,
                seed=seed,
                device=device,
                epochs=epochs,
                patience=patience,
                learning_rate=learning_rate,
            )
            save_csv(seed_checkpoints.assign(seed=seed), run_dir / f"seed_{seed}_checkpoints.csv")
            nodes, _ = graph_node_metrics(retained)
        if seed == SEED:
            save_csv(
                pd.DataFrame(
                    [
                        {
                            "seed": seed,
                            "target_index": row.target_index,
                            "best_epoch": row.best_epoch,
                            "best_val_mse": row.best_val_mse,
                            "epochs_run": row.epochs_run,
                        }
                        for row in pd.read_csv(
                            directories["tables"] / "ngc_checkpoints_seed42.csv"
                        ).itertuples(index=False)
                    ]
                ),
                run_dir / f"seed_{seed}_checkpoints.csv",
            )
        retained_runs.append(retained > 0)
        importance_runs.append(importance)
        signed = np.zeros_like(importance)
        for row in audit.itertuples(index=False):
            signed[DISPLAY_LABELS.index(row.source), DISPLAY_LABELS.index(row.target)] = row.signed_relative_loss_change
        signed_runs.append(signed)
        roles = nodes.set_index("node")
        for node in DISPLAY_LABELS:
            role_runs.append(
                {
                    "seed": seed,
                    "node": node,
                    "role": roles.loc[node, "role"],
                    "net_strength": roles.loc[node, "net_strength"],
                }
            )
        reference_edges = main_result["retained"] > 0
        current_edges = retained > 0
        mask = ~np.eye(len(DISPLAY_LABELS), dtype=bool)
        intersection = np.sum(reference_edges & current_edges & mask)
        union = np.sum((reference_edges | current_edges) & mask)
        run_summaries.append(
            {
                "seed": seed,
                "q70_threshold": threshold,
                "num_retained_edges": int(np.sum(current_edges & mask)),
                "jaccard_with_seed42": float(intersection / union) if union else 1.0,
            }
        )
        per_seed = audit.copy()
        per_seed["seed"] = seed
        save_csv(per_seed, run_dir / f"seed_{seed}_importance.csv")

    selected = np.stack(retained_runs).astype(float)
    positive = np.stack(importance_runs)
    signed = np.stack(signed_runs)
    rows = []
    for source, source_label in enumerate(DISPLAY_LABELS):
        for target, target_label in enumerate(DISPLAY_LABELS):
            if source == target:
                continue
            frequency = float(selected[:, source, target].mean())
            rows.append(
                {
                    "source": source_label,
                    "target": target_label,
                    "selection_frequency": frequency,
                    "mean_positive_importance": float(positive[:, source, target].mean()),
                    "std_positive_importance": float(positive[:, source, target].std(ddof=1)) if len(seed_values) > 1 else 0.0,
                    "mean_signed_loss_change": float(signed[:, source, target].mean()),
                    "std_signed_loss_change": float(signed[:, source, target].std(ddof=1)) if len(seed_values) > 1 else 0.0,
                    "stable": frequency >= CAMERA_STABILITY_THRESHOLD,
                }
            )
    edge_stability = pd.DataFrame(rows).sort_values(
        ["selection_frequency", "mean_positive_importance"], ascending=False
    )

    role_frame = pd.DataFrame(role_runs)
    role_summary = (
        role_frame.groupby("node")
        .agg(
            transmitter_frequency=("role", lambda x: float(np.mean(x == "Transmitter"))),
            receiver_frequency=("role", lambda x: float(np.mean(x == "Receiver"))),
            mean_net_strength=("net_strength", "mean"),
            std_net_strength=("net_strength", "std"),
        )
        .reset_index()
    )
    run_summary = pd.DataFrame(run_summaries)
    save_csv(edge_stability, directories["tables"] / "edge_stability_20seeds.csv")
    save_csv(role_summary, directories["tables"] / "node_role_stability_20seeds.csv")
    save_csv(run_summary, directories["tables"] / "stability_run_summary.csv")
    return edge_stability, role_summary, run_summary


def summarize_variant(
    name: str,
    split: SupervisedSplit,
    retained: np.ndarray,
    threshold: float,
    nodes: pd.DataFrame,
) -> dict:
    vni = DISPLAY_LABELS.index("VNI")
    incoming = [DISPLAY_LABELS[i] for i in range(len(DISPLAY_LABELS)) if retained[i, vni] > 0]
    vni_row = nodes.set_index("node").loc["VNI"]
    transmitters = nodes.loc[nodes["role"].eq("Transmitter")].nlargest(3, "net_strength")["node"].tolist()
    return {
        "variant": name,
        "raw_observations": len(split.raw_dates),
        "train_forecasts": len(split.X_train),
        "validation_forecasts": len(split.X_val),
        "test_forecasts": len(split.X_test),
        "VNI_incoming_edges": ",".join(incoming),
        "VNI_in_strength": float(vni_row["in_strength"]),
        "VNI_out_strength": float(vni_row["out_strength"]),
        "VNI_net_strength": float(vni_row["net_strength"]),
        "VNI_role": vni_row["role"],
        "top_transmitters": ",".join(transmitters),
        "q70_threshold": threshold,
        "num_edges": int(np.count_nonzero(retained)),
    }


def run_robustness(
    main_returns: pd.DataFrame,
    common_returns: pd.DataFrame,
    main_split: SupervisedSplit,
    main_result: dict,
    directories: dict[str, Path],
    *,
    device: str,
    epochs: int,
    patience: int,
    learning_rate: float,
) -> pd.DataFrame:
    print("\n=== Robustness: common trading days ===")
    common_split = prepare_supervised(common_returns, CAMERA_LAGS)
    _, _, _, common_importance, common_retained, common_threshold, common_audit = train_variant_ngc(
        common_split,
        seed=SEED,
        device=device,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
    )
    common_nodes, common_graph = graph_node_metrics(common_retained)
    save_csv(common_audit, directories["tables"] / "common_day_ablation_importance.csv")
    save_csv(common_nodes, directories["tables"] / "common_day_node_roles.csv")
    save_csv(matrix_frame(common_retained), directories["tables"] / "common_day_retained_adjacency.csv")
    draw_prediction_graph(
        common_graph,
        directories["figures"] / "common_day_prediction_graph.png",
        "Common-trading-day prediction network",
    )
    common_var, common_var_adj, _ = fit_var_granger(common_returns, common_split.val_end, CAMERA_LAGS)
    save_csv(common_var, directories["tables"] / "common_day_var20_granger.csv")
    write_json(
        compare_binary_graphs(common_retained, common_var_adj),
        directories["tables"] / "common_day_var_ngc_comparison.json",
    )

    print("\n=== Robustness: conservative extra shift for global-market inputs ===")
    shifted_sources = (
        "return_oil",
        "return_gold",
        "return_sp500",
        "return_vix",
    )
    shifted_split = prepare_supervised(
        main_returns, CAMERA_LAGS, input_shift_sources=shifted_sources
    )
    _, _, _, shifted_importance, shifted_retained, shifted_threshold, shifted_audit = train_variant_ngc(
        shifted_split,
        seed=SEED,
        device=device,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
    )
    shifted_nodes, shifted_graph = graph_node_metrics(shifted_retained)
    save_csv(shifted_audit, directories["tables"] / "timing_shift_ablation_importance.csv")
    save_csv(shifted_nodes, directories["tables"] / "timing_shift_node_roles.csv")
    save_csv(matrix_frame(shifted_retained), directories["tables"] / "timing_shift_retained_adjacency.csv")
    draw_prediction_graph(
        shifted_graph,
        directories["figures"] / "timing_shift_prediction_graph.png",
        "Conservative timing-shift prediction network",
    )

    main_summary = summarize_variant(
        "main_calendar",
        main_split,
        main_result["retained"],
        main_result["threshold"],
        main_result["node_metrics"],
    )
    common_summary = summarize_variant(
        "common_trading_days_no_interpolation",
        common_split,
        common_retained,
        common_threshold,
        common_nodes,
    )
    shifted_summary = summarize_variant(
        "main_calendar_extra_global_input_shift",
        shifted_split,
        shifted_retained,
        shifted_threshold,
        shifted_nodes,
    )
    robustness = pd.DataFrame([main_summary, common_summary, shifted_summary])
    save_csv(robustness, directories["tables"] / "calendar_timing_robustness.csv")
    save_split_summary(common_split, directories["tables"] / "common_day_split_summary.csv", "common_day")
    save_split_summary(shifted_split, directories["tables"] / "timing_shift_split_summary.csv", "timing_shift")
    return robustness


def write_results_summary(
    path: Path,
    split: SupervisedSplit,
    main_result: dict,
    stability: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None,
    robustness: pd.DataFrame | None,
) -> None:
    metrics = main_result["forecast_metrics"].set_index("model")
    nodes = main_result["node_metrics"].set_index("node")
    inference = main_result["inference"]
    lines = [
        "# Camera-ready actual-results summary",
        "",
        "This file is generated from completed runs. It is the numeric source of truth for manuscript revision.",
        "",
        "## Experiment contract",
        "",
        f"- Split: {json.dumps(split.split_summary, ensure_ascii=False)}",
        f"- NGC architecture: `{CAMERA_LAGS * len(RETURN_COLS)}-{CAMERA_NGC_HIDDEN[0]}-{CAMERA_NGC_HIDDEN[1]}-1`.",
        f"- Graph threshold: validation block-ablation positive-importance q{CAMERA_GRAPH_QUANTILE:.2f} = {main_result['threshold']:.8f}.",
        "- Validation is used for checkpoint selection and exploratory graph attribution; the test set is untouched until forecast evaluation.",
        "",
        "## Held-out VNI forecasting",
        "",
        metrics.reset_index().to_markdown(index=False),
        "",
        "## Forecast inference",
        "",
        "```json",
        json.dumps(inference, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Prediction-weighted node roles",
        "",
        nodes.reset_index().to_markdown(index=False),
        "",
        "## VAR-NGC graph comparison",
        "",
        "```json",
        json.dumps(main_result["graph_comparison"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## GBRI regime distribution",
        "",
        main_result["gbri_distribution"].to_markdown(index=False),
    ]
    if stability is not None:
        edges, roles, runs = stability
        lines.extend(
            [
                "",
                "## Multi-seed stability",
                "",
                f"- Seeds completed: {len(runs)}.",
                f"- Stable edges (selection frequency >= {CAMERA_STABILITY_THRESHOLD:.2f}): {int(edges['stable'].sum())}.",
                "",
                roles.to_markdown(index=False),
            ]
        )
    if robustness is not None:
        lines.extend(["", "## Calendar and timing robustness", "", robustness.to_markdown(index=False)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    directories = output_directories(args.smoke_test)
    device = resolve_device(args.device)
    epochs = 3 if args.smoke_test else args.epochs
    patience = 2 if args.smoke_test else args.patience
    stability_count = min(2, args.stability_seeds) if args.smoke_test else args.stability_seeds
    seed_values = list(CAMERA_STABILITY_SEEDS[:stability_count])
    print(f"Device: {device}; epochs={epochs}; patience={patience}; stability_seeds={seed_values}")

    main_prices, main_returns, main_fill_audit, main_invalid = build_return_dataset("main")
    common_prices, common_returns, common_fill_audit, common_invalid = build_return_dataset("common")
    save_csv(main_prices, directories["data"] / "main_price_panel.csv")
    save_csv(main_returns, directories["data"] / "main_return_panel.csv")
    save_csv(common_prices, directories["data"] / "common_day_price_panel.csv")
    save_csv(common_returns, directories["data"] / "common_day_return_panel.csv")
    save_csv(pd.concat([main_fill_audit, common_fill_audit]), directories["tables"] / "missing_value_audit.csv")
    invalid = pd.concat(
        [main_invalid.assign(calendar="main"), common_invalid.assign(calendar="common")],
        ignore_index=True,
    )
    save_csv(invalid, directories["tables"] / "invalid_return_endpoints.csv")
    write_data_audit(
        directories["root"] / "data_audit.md",
        main_prices,
        main_returns,
        common_returns,
        main_invalid,
    )

    split = prepare_supervised(main_returns, CAMERA_LAGS)
    save_split_summary(split, directories["tables"] / "main_split_summary.csv", "main")
    config = {
        "lag": CAMERA_LAGS,
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
        "ngc_hidden": list(CAMERA_NGC_HIDDEN),
        "lstm_hidden": CAMERA_LSTM_HIDDEN,
        "ngc_lambda": CAMERA_NGC_LAMBDA,
        "graph_quantile": CAMERA_GRAPH_QUANTILE,
        "gbri_alpha": CAMERA_GBRI_ALPHA,
        "volatility_window": CAMERA_VOL_WINDOW,
        "stability_seeds": seed_values,
        "stability_threshold": CAMERA_STABILITY_THRESHOLD,
        "device": device,
        "epochs": epochs,
        "patience": patience,
        "learning_rate": args.learning_rate,
        "forecast_origin": "Before VNI opens on target date t; inputs end at date t-1.",
        "timing_sensitivity": "OIL/GOLD/SP500/VIX inputs receive one additional observation shift.",
    }
    write_json(config, directories["root"] / "experiment_config.json")

    main_result = run_main_models(
        split,
        main_returns,
        directories,
        device=device,
        epochs=epochs,
        patience=patience,
        learning_rate=args.learning_rate,
    )

    stability = None
    if not args.skip_stability:
        stability = run_stability(
            split,
            directories,
            main_result,
            seed_values=seed_values,
            device=device,
            epochs=epochs,
            patience=patience,
            learning_rate=args.learning_rate,
        )

    robustness = None
    if not args.skip_robustness:
        robustness = run_robustness(
            main_returns,
            common_returns,
            split,
            main_result,
            directories,
            device=device,
            epochs=epochs,
            patience=patience,
            learning_rate=args.learning_rate,
        )

    write_results_summary(
        directories["root"] / "camera_ready_results_summary.md",
        split,
        main_result,
        stability,
        robustness,
    )
    print(f"\nCamera-ready pipeline completed: {directories['root']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leakage-aware camera-ready revision pipeline")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--stability-seeds", type=int, default=20, choices=range(1, 21))
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    main(parser.parse_args())
