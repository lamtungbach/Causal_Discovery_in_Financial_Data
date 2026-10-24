from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.seed_uncertainty import run_analysis  # noqa: E402


def main() -> None:
    output_dir = ROOT / "results" / "journal_revision" / "seed_uncertainty"
    result = run_analysis(output_dir)
    selected_80 = result["consensus"].loc[
        result["consensus"]["consensus_level"].eq(0.80)
        & result["consensus"]["selected_in_consensus"]
    ]
    print(f"PASS: seed uncertainty written to {output_dir}")
    print(f"Saved seeds analyzed: {len(result['graphs'])} ({min(result['graphs'])}-{max(result['graphs'])})")
    print(f"Pairwise graph comparisons: {len(result['pairwise'])}")
    print(f"Edges in 0.80 consensus graph: {len(selected_80)}")
    for row in selected_80.itertuples(index=False):
        print(f"  {row.source} -> {row.target}: {row.selection_frequency:.2f}")


if __name__ == "__main__":
    main()
