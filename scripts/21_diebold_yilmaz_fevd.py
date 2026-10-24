from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fevd_connectedness import HORIZONS, run_fevd


def main() -> None:
    output_dir = ROOT / "results" / "journal_revision" / "fevd"
    result = run_fevd(output_dir)
    systems = result["summary"].loc[result["summary"]["record_type"].eq("system")]
    print(f"PASS: generalized FEVD outputs written to {output_dir}")
    for row in systems.itertuples(index=False):
        print(f"H={row.horizon}: TCI={row.total_connectedness_index:.3f}%")
    print(f"Horizons produced: {', '.join(map(str, HORIZONS))}")
    print(f"Ordering checks passed: {int(result['checks']['passed_tolerance_1e_10'].sum())}/4")


if __name__ == "__main__":
    main()
