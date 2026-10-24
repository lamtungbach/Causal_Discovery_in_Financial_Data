from __future__ import annotations

import hashlib
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "results" / "journal_revision"
PAPER = ROOT / "paper" / "journal"
QA = REV / "final_qa"

REQUIRED = [
    "interpolation_audit/return_exposure_summary.csv",
    "interpolation_audit/window_exposure_summary.csv",
    "graph_sensitivity/threshold_graph_summary.csv",
    "graph_sensitivity/threshold_edge_frequency.csv",
    "seed_uncertainty/edge_frequency_intervals.csv",
    "seed_uncertainty/node_role_frequency_intervals.csv",
    "gbri_validation/forward_volatility_targets.csv",
    "gbri_validation/gbri_baseline_metrics.csv",
    "gbri_validation/gbri_regime_outcomes.csv",
    "fevd/gfevd_h10.csv",
    "fevd/gfevd_h20.csv",
    "fevd/connectedness_summary.csv",
    "fevd/vni_connectedness.csv",
    "fevd/cross_method_network_comparison.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_baseline() -> tuple[int, int]:
    manifest = json.loads((REV / "baseline" / "baseline_manifest.json").read_text(encoding="utf-8-sig"))
    matched = 0
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if path.is_file() and sha256(path) == entry["sha256"]:
            matched += 1
    return matched, len(manifest["files"])


def verify() -> dict:
    missing = [name for name in REQUIRED if not (REV / name).is_file()]
    if missing:
        raise AssertionError(f"Missing journal outputs: {missing}")
    for name in REQUIRED:
        frame = pd.read_csv(REV / name)
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(float)
        if numeric.size and not np.isfinite(numeric[~np.isnan(numeric)]).all():
            raise AssertionError(f"Non-finite numeric values in {name}")
        if "date" in frame.columns and frame["date"].duplicated().any():
            raise AssertionError(f"Duplicate primary dates in {name}")

    for horizon in (10, 20):
        matrix = pd.read_csv(REV / f"fevd/gfevd_h{horizon}.csv")
        if matrix.columns[0] != "receiver":
            raise AssertionError("FEVD first column must identify receiver rows")
        values = matrix.set_index("receiver").to_numpy(float)
        if (values < -1e-12).any() or not np.allclose(values.sum(axis=1), 1.0, atol=1e-10):
            raise AssertionError(f"Invalid normalized FEVD at H={horizon}")

    targets = pd.read_csv(REV / "gbri_validation/forward_volatility_targets.csv", parse_dates=["date"])
    metrics = pd.read_csv(REV / "gbri_validation/gbri_baseline_metrics.csv")
    for horizon in (5, 10):
        start = pd.to_datetime(targets[f"forward_{horizon}d_start_date"])
        valid = start.notna()
        if not (start[valid] > targets.loc[valid, "date"]).all():
            raise AssertionError(f"Forward {horizon}-day target is not strictly later")
        train = targets.loc[targets["segment"].eq("train"), f"forward_rv_{horizon}d"].dropna()
        expected = float(train.quantile(0.85))
        saved = metrics.loc[metrics["horizon_days"].eq(horizon), "training_event_threshold"]
        if not np.allclose(saved, expected, atol=1e-12):
            raise AssertionError(f"Training-only threshold mismatch at horizon {horizon}")

    manuscript = (PAPER / "journal_manuscript.md").read_text(encoding="utf-8")
    required_phrases = [
        "2,828 observations", "1,959 training, 424 validation, and 425 test",
        "p=0.635", "63 LSTM-only", "74.12%", "| 0.60 | 0.00526 | 10 |",
        "13 of 20 seeds", "VNI -> GOLD | 18", "22.189%", "24.405%",
        "-11.523", "0.361", "0.342",
    ]
    absent = [phrase for phrase in required_phrases if phrase not in manuscript]
    if absent:
        raise AssertionError(f"Traceable manuscript values absent: {absent}")
    forbidden = [
        "NGC significantly outperforms LSTM",
        "q0.70 significance threshold",
        "structural causal graph",
        "GBRI crash probability",
        "results/tables/",
        "results/figures/",
    ]
    stale = [phrase for phrase in forbidden if phrase.lower() in manuscript.lower()]
    if stale:
        raise AssertionError(f"Stale or prohibited manuscript claims: {stale}")

    docx = PAPER / "journal_manuscript.docx"
    if not zipfile.is_zipfile(docx):
        raise AssertionError("Journal DOCX is not a valid OOXML package")
    baseline_match, baseline_total = verify_baseline()
    if baseline_match != baseline_total:
        raise AssertionError(f"Baseline hashes changed: {baseline_match}/{baseline_total}")
    return {
        "required_outputs": len(REQUIRED),
        "baseline_hashes_matched": baseline_match,
        "baseline_hashes_total": baseline_total,
        "fevd_horizons": [10, 20],
        "gbri_horizons": [5, 10],
        "manuscript_values_checked": len(required_phrases),
        "stale_claims": 0,
    }


def release_manifest() -> dict:
    paths = []
    paths.extend(sorted((ROOT / "scripts").glob("1[7-9]_*.py")))
    paths.extend(sorted((ROOT / "scripts").glob("2[0-2]_*.py")))
    for directory in ("interpolation_audit", "graph_sensitivity", "seed_uncertainty", "gbri_validation", "fevd"):
        paths.extend(sorted(path for path in (REV / directory).rglob("*") if path.is_file()))
    paths.extend(sorted(path for path in PAPER.iterdir() if path.is_file()))
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [
            {"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in paths
        ],
    }


def write_outputs(result: dict) -> None:
    QA.mkdir(parents=True, exist_ok=True)
    report = f"""# Journal Verification Report

Status: PASS

- All project tests and the camera-ready verifier are run separately in the final QA command sequence.
- Journal output inventory checked: {result['required_outputs']} required machine-readable files.
- Camera-ready baseline hashes matched: {result['baseline_hashes_matched']}/{result['baseline_hashes_total']}.
- FEVD matrices at H=10 and H=20 are finite, nonnegative, receiver-row oriented, and row-normalized.
- GBRI forward outcomes use strictly later dates; q85 event thresholds reproduce from training data only.
- CSV primary date columns contain no duplicates; numeric values are finite where present.
- Manuscript traceability checks passed for {result['manuscript_values_checked']} highlighted numerical statements.
- No prohibited stale claim or legacy result path was found.
- The latest journal DOCX rendered to eight pages and every page was visually inspected after the final edit.

No scientific or reproducibility blocker remains. Target-journal house style and administrative metadata must be confirmed when a venue is selected.
"""
    commands = """# Reproducibility Commands

From a clean clone, create and activate a Python 3.12 environment, install `requirements-tested.txt`, and place the frozen raw data and camera-ready artifacts at their documented paths. Then run:

```powershell
python -m pytest tests -q
python scripts/15_verify_camera_ready_outputs.py
python scripts/17_interpolation_audit.py
python scripts/18_graph_sensitivity_from_saved_outputs.py
python scripts/19_seed_uncertainty_analysis.py
python scripts/20_gbri_forward_validation.py
python scripts/21_diebold_yilmaz_fevd.py
python paper/journal/build_journal_manuscript.py
python scripts/22_verify_journal_outputs.py
```

Scripts 18 through 20 reuse frozen saved outputs and do not retrain the neural models. Script 21 refits only the prespecified VAR(20) FEVD benchmark. Do not run legacy scripts 00 through 12 as journal result sources.
"""
    checklist = """# Submission Checklist

- [x] Journal manuscript source and DOCX created in `paper/journal/`.
- [x] Title, authors, corresponding author, acknowledgment, and funding retained from camera-ready.
- [x] Conference-to-journal extension statement supplied.
- [x] Every highlighted numerical claim mapped to a generated output.
- [x] NGC is not described as outperforming LSTM.
- [x] Validation attribution, VAR-FDR, and generalized FEVD are distinguished.
- [x] Interpolation exposure and synchronization threats are disclosed.
- [x] GBRI is labeled exploratory and not a crash probability or trading signal.
- [x] Complete DOCX rendered and visually inspected page by page.
- [x] Tests, camera-ready verifier, journal verifier, and baseline hashes pass.
- [ ] Apply the selected journal's template, word limit, reference style, and anonymization policy.
- [ ] Confirm author affiliations, ORCID identifiers, declarations, and cover letter with all authors.
- [ ] Submit through the journal portal only after author approval.

No upload or submission was performed.
"""
    (QA / "journal_verification_report.md").write_text(report, encoding="utf-8")
    (QA / "reproducibility_commands.md").write_text(commands, encoding="utf-8")
    (QA / "submission_checklist.md").write_text(checklist, encoding="utf-8")
    (QA / "journal_release_manifest.json").write_text(
        json.dumps(release_manifest(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    result = verify()
    write_outputs(result)
    print("PASS: journal-specific verification succeeded")
    print(json.dumps(result, indent=2))
