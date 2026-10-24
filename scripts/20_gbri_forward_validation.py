from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gbri_validation import run_validation  # noqa: E402


def main() -> None:
    output_dir = ROOT / "results" / "journal_revision" / "gbri_validation"
    result = run_validation(output_dir)
    print(f"PASS: GBRI validation written to {output_dir}")
    for horizon, threshold in result["thresholds"].items():
        rows = result["metrics"].loc[result["metrics"]["horizon_days"].eq(horizon)]
        gbri = rows.loc[rows["model"].eq("GBRI")].iloc[0]
        print(
            f"h={horizon}: training q85 threshold={threshold:.6f}, test n={gbri.test_n}, "
            f"GBRI Spearman={gbri.spearman:.3f}, ROC-AUC={gbri.roc_auc:.3f}, "
            f"PR-AUC={gbri.pr_auc:.3f}, Brier={gbri.brier:.3f}"
        )


if __name__ == "__main__":
    main()
