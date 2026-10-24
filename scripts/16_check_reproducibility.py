"""Verify deterministic repetition and non-degenerate seed sensitivity.

The same-seed comparison is expected to be exactly zero on a fixed software
and hardware stack. A separate seed must not be identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CAMERA_LAGS,
    CAMERA_NGC_HIDDEN,
    CAMERA_NGC_LAMBDA,
    CAMERA_TABLE_DIR,
)
from src.camera_ready_data import build_return_dataset, prepare_supervised  # noqa: E402
from src.camera_ready_models import predict_ngc, resolve_device, train_ngc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check NGC seed reproducibility")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def fit_predictions(split, seed: int, device: str, args: argparse.Namespace) -> np.ndarray:
    models, _, _ = train_ngc(
        split.X_train,
        split.Y_train,
        split.X_val,
        split.Y_val,
        hidden=CAMERA_NGC_HIDDEN,
        lam=CAMERA_NGC_LAMBDA,
        seed=seed,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.learning_rate,
    )
    return predict_ngc(models, split.X_test)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    _, returns, _, _ = build_return_dataset("main")
    split = prepare_supervised(returns, CAMERA_LAGS)

    seed42_first = fit_predictions(split, 42, device, args)
    seed42_repeat = fit_predictions(split, 42, device, args)
    seed43 = fit_predictions(split, 43, device, args)

    same_seed_diff = float(np.max(np.abs(seed42_first - seed42_repeat)))
    different_seed_diff = float(np.max(np.abs(seed42_first - seed43)))
    prediction_std = float(np.std(seed42_first))
    target_std = float(np.std(split.Y_test))

    if same_seed_diff > 1e-10:
        raise AssertionError(f"Same-seed run is not reproducible: max diff={same_seed_diff}")
    if different_seed_diff <= 1e-10:
        raise AssertionError("Different seeds produced indistinguishable predictions")
    if prediction_std <= 1e-10 or target_std <= 1e-10:
        raise AssertionError("Prediction or target data are degenerate constants")

    report = {
        "device": device,
        "seed_repeated": 42,
        "comparison_seed": 43,
        "max_abs_repeat_diff": same_seed_diff,
        "max_abs_different_seed_diff": different_seed_diff,
        "seed42_prediction_std": prediction_std,
        "test_target_std_standardized": target_std,
        "interpretation": (
            "A zero same-seed difference is expected under deterministic execution; "
            "the nonzero different-seed difference and nonzero standard deviations "
            "rule out a constant-output explanation."
        ),
    }
    CAMERA_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output = CAMERA_TABLE_DIR / "reproducibility_check.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"PASS: wrote {output}")


if __name__ == "__main__":
    main()
