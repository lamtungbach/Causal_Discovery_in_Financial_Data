"""Canonical entry point for the KSE 2026 camera-ready experiments."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LEGACY_SCRIPTS = [
    "scripts/00_check_environment.py",
    "scripts/01_validate_data.py",
    "scripts/02_build_features.py",
    "scripts/03_eda_stationarity.py",
    "scripts/04_modwt_multiscale.py",
    "scripts/05_wavelet_coherence.py",
    "scripts/06_var_granger_baseline.py",
    "scripts/07_lstm_forecasting_baseline.py",
    "scripts/08_neural_granger.py",
    "scripts/08b_neural_granger_multiband.py",
    "scripts/09_graph_analysis.py",
    "scripts/10_event_window_analysis.py",
    "scripts/11_graph_risk_index.py",
    "scripts/12_ablation_robustness.py",
]


def run(command: list[str]) -> None:
    print(f"\nRUN {' '.join(command)}", flush=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(command, cwd=ROOT, check=True, env=environment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the KSE camera-ready pipeline (default) or the archived legacy pipeline."
    )
    parser.add_argument(
        "--profile",
        choices=("camera-ready", "smoke", "legacy"),
        default="camera-ready",
        help="camera-ready is authoritative; smoke is a fast wiring check.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--stability-seeds",
        type=int,
        default=20,
        help="The full camera-ready verifier requires 20; use profile=smoke for a short run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile == "legacy":
        print("WARNING: running archived scripts 00-12; these are not camera-ready results.")
        for script in LEGACY_SCRIPTS:
            run([sys.executable, script])
        print("\nLegacy pipeline completed")
        return

    command = [
        sys.executable,
        "scripts/14_camera_ready_revision.py",
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--stability-seeds",
        str(args.stability_seeds),
    ]
    if args.profile == "smoke":
        command.append("--smoke-test")
    run(command)

    if args.profile == "camera-ready":
        run([sys.executable, "scripts/15_verify_camera_ready_outputs.py"])
        print("\nCamera-ready pipeline and integrity verification completed")
    else:
        print("\nSmoke test completed; full-output verification is intentionally not run")


if __name__ == "__main__":
    main()
