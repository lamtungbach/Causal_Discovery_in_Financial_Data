from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interpolation_audit import run_audit  # noqa: E402


def main() -> None:
    output_dir = ROOT / "results" / "journal_revision" / "interpolation_audit"
    result = run_audit(output_dir)
    print(f"PASS: interpolation audit written to {output_dir}")
    print(f"Raw CSV hashes preserved: {len(result['raw_hashes'])}")
    print(f"Canonical maximum absolute difference: {result['max_abs_difference']:.3g}")
    print(f"Audited price cells: {result['price_cells']}")
    print(f"Linearly interpolated cells: {result['interpolated_cells']}")


if __name__ == "__main__":
    main()
