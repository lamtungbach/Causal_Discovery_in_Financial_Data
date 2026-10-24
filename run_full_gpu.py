"""Compatibility wrapper for a full camera-ready CUDA run.

New commands should use ``python run_pipeline.py --device cuda`` directly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    command = [
        sys.executable,
        "run_pipeline.py",
        "--profile",
        "camera-ready",
        "--device",
        "cuda",
        "--epochs",
        "300",
        "--patience",
        "30",
        "--stability-seeds",
        "20",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
