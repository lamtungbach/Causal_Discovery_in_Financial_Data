from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph_sensitivity import run_analysis  # noqa: E402


def main() -> None:
    output_dir = ROOT / "results" / "journal_revision" / "graph_sensitivity"
    result = run_analysis(output_dir)
    seed42 = result["graph_summary"].loc[result["graph_summary"]["seed"].eq(42)]
    print(f"PASS: graph sensitivity written to {output_dir}")
    print(f"Saved seeds analyzed: {len(result['seeds'])} ({min(result['seeds'])}-{max(result['seeds'])})")
    for row in seed42.itertuples(index=False):
        print(
            f"q{int(row.quantile * 100)} seed-42: threshold={row.threshold:.12g}, "
            f"edges={row.edge_count}, VNI role={row.vni_role}, "
            f"incoming={row.incoming_vni_sources or 'none'}"
        )


if __name__ == "__main__":
    main()
