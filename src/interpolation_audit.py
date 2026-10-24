from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from config import ASSETS, CAMERA_LAGS, CAMERA_READY_DIR, RAW_DIR, RETURN_COLS
from src.camera_ready_data import (
    RAW_FILES,
    build_price_panel,
    load_raw_close,
    prepare_supervised,
    safe_log_return_panel,
)


ORIGINAL = "originally_observed"
FORWARD_FILLED = "forward_filled"
LINEAR_INTERPOLATED = "linearly_interpolated"
MISSING_DROPPED = "missing_dropped"
FILL_LABELS = {FORWARD_FILLED, LINEAR_INTERPOLATED}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_csv_hashes() -> dict[str, str]:
    return {
        path.relative_to(RAW_DIR).as_posix(): sha256_file(path)
        for path in sorted(RAW_DIR.glob("*.csv"))
    }


def build_unfilled_main_panel() -> pd.DataFrame:
    """Recreate the VN-Index-anchored left joins before any filling."""
    frames = {asset: load_raw_close(asset) for asset in RAW_FILES}
    panel = frames["vnindex"].copy()
    for asset in RAW_FILES:
        if asset != "vnindex":
            panel = panel.merge(frames[asset], on="date", how="left")
    return panel


def apply_main_fill_with_provenance(
    aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the camera-ready fill order and retain a label for every cell."""
    filled = aligned.copy()
    price_cols = [meta["price"] for meta in ASSETS.values()]
    provenance = pd.DataFrame({"date": filled["date"]})
    for col in price_cols:
        provenance[col] = np.where(filled[col].notna(), ORIGINAL, MISSING_DROPPED)

    for asset in ("oil", "gold", "sp500", "vix"):
        col = ASSETS[asset]["price"]
        before = filled[col].copy()
        after = before.ffill(limit=3)
        mask = before.isna() & after.notna()
        provenance.loc[mask, col] = FORWARD_FILLED
        filled[col] = after

    sse_col = ASSETS["sse"]["price"]
    before_sse = filled[sse_col].copy()
    after_ffill = before_sse.ffill(limit=3)
    ffill_mask = before_sse.isna() & after_ffill.notna()
    provenance.loc[ffill_mask, sse_col] = FORWARD_FILLED

    after_interpolation = after_ffill.interpolate(method="linear", limit=5)
    interpolation_mask = after_ffill.isna() & after_interpolation.notna()
    provenance.loc[interpolation_mask, sse_col] = LINEAR_INTERPOLATED
    filled[sse_col] = after_interpolation

    return filled, provenance


def reconstruct_main_panel_with_provenance(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aligned = build_unfilled_main_panel()
    filled, provenance_wide = apply_main_fill_with_provenance(aligned)
    price_cols = [meta["price"] for meta in ASSETS.values()]
    row_kept = filled[price_cols].notna().all(axis=1)
    final_panel = filled.loc[row_kept, ["date", *price_cols]].reset_index(drop=True)

    records: list[pd.DataFrame] = []
    for asset, meta in ASSETS.items():
        col = meta["price"]
        records.append(
            pd.DataFrame(
                {
                    "date": aligned["date"],
                    "asset": asset,
                    "price_column": col,
                    "raw_aligned_price": aligned[col],
                    "price_after_policy": filled[col],
                    "provenance": provenance_wide[col],
                    "row_kept": row_kept,
                }
            )
        )
    provenance_long = pd.concat(records, ignore_index=True).sort_values(
        ["date", "asset"]
    )
    return final_panel, provenance_long.reset_index(drop=True), aligned


def assert_matches_canonical(reconstructed: pd.DataFrame) -> float:
    canonical_path = CAMERA_READY_DIR / "data" / "main_price_panel.csv"
    canonical = pd.read_csv(canonical_path, parse_dates=["date"])
    if reconstructed.columns.tolist() != canonical.columns.tolist():
        raise AssertionError("Reconstructed price-panel columns differ from canonical output")
    if not reconstructed["date"].equals(canonical["date"]):
        raise AssertionError("Reconstructed price-panel dates differ from canonical output")
    cols = [meta["price"] for meta in ASSETS.values()]
    delta = np.abs(
        reconstructed[cols].to_numpy(dtype=float)
        - canonical[cols].to_numpy(dtype=float)
    )
    max_abs_difference = float(np.nanmax(delta)) if delta.size else 0.0
    if not np.allclose(
        reconstructed[cols].to_numpy(dtype=float),
        canonical[cols].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    ):
        raise AssertionError(
            "Reconstructed panel differs from canonical saved panel; "
            f"maximum absolute difference={max_abs_difference}"
        )

    current, _ = build_price_panel("main")
    pd.testing.assert_frame_equal(reconstructed, current, check_exact=True)
    return max_abs_difference


def _split_lookup(returns: pd.DataFrame) -> tuple[dict[pd.Timestamp, str], object]:
    split = prepare_supervised(returns, CAMERA_LAGS)
    lookup: dict[pd.Timestamp, str] = {}
    for name, dates in (
        ("train", split.dates_train),
        ("validation", split.dates_val),
        ("test", split.dates_test),
    ):
        lookup.update({pd.Timestamp(date): name for date in dates})
    return lookup, split


def build_return_exposure_detail(
    price_panel: pd.DataFrame,
    provenance_long: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, object]:
    kept = provenance_long.loc[provenance_long["row_kept"]].copy()
    wide = kept.pivot(index="date", columns="asset", values="provenance")
    wide = wide.reindex(pd.DatetimeIndex(price_panel["date"]))
    split_lookup, split = _split_lookup(returns)
    valid_return_dates = set(pd.to_datetime(returns["date"]))

    rows: list[dict] = []
    for asset in ASSETS:
        labels = wide[asset].to_numpy(dtype=object)
        for i in range(1, len(price_panel)):
            date = pd.Timestamp(price_panel.loc[i, "date"])
            if date not in valid_return_dates:
                continue
            previous_label = labels[i - 1]
            current_label = labels[i]
            forward = previous_label == FORWARD_FILLED or current_label == FORWARD_FILLED
            linear = (
                previous_label == LINEAR_INTERPOLATED
                or current_label == LINEAR_INTERPOLATED
            )
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "split": split_lookup.get(date),
                    "previous_price_provenance": previous_label,
                    "current_price_provenance": current_label,
                    "forward_fill_exposed": bool(forward),
                    "linear_interpolation_exposed": bool(linear),
                    "any_fill_exposed": bool(forward or linear),
                }
            )
    return pd.DataFrame(rows), split


def _percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def summarize_return_exposure(detail: pd.DataFrame) -> pd.DataFrame:
    eligible = detail.loc[detail["split"].notna()].copy()
    rows: list[dict] = []
    for split_name in ("train", "validation", "test"):
        split_frame = eligible.loc[eligible["split"].eq(split_name)]
        for asset in [*ASSETS.keys(), "ALL"]:
            frame = split_frame if asset == "ALL" else split_frame.loc[split_frame["asset"].eq(asset)]
            endpoints = len(frame)
            exposed = int(frame["any_fill_exposed"].sum())
            forward = int(frame["forward_fill_exposed"].sum())
            linear = int(frame["linear_interpolation_exposed"].sum())
            rows.append(
                {
                    "split": split_name,
                    "asset": asset,
                    "target_dates": int(frame["date"].nunique()),
                    "return_endpoint_cells": endpoints,
                    "exposed_return_cells": exposed,
                    "exposed_return_cells_pct": _percentage(exposed, endpoints),
                    "forward_fill_exposed_cells": forward,
                    "linear_interpolation_exposed_cells": linear,
                    "target_dates_with_any_exposure": int(
                        frame.loc[frame["any_fill_exposed"], "date"].nunique()
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_window_exposure(
    returns: pd.DataFrame,
    return_detail: pd.DataFrame,
    split: object,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(returns["date"]))
    assets = list(ASSETS)
    detail_indexed = return_detail.set_index(["date", "asset"])
    any_matrix = np.zeros((len(dates), len(assets)), dtype=bool)
    forward_matrix = np.zeros_like(any_matrix)
    linear_matrix = np.zeros_like(any_matrix)
    for i, date in enumerate(dates):
        for j, asset in enumerate(assets):
            row = detail_indexed.loc[(date, asset)]
            any_matrix[i, j] = bool(row["any_fill_exposed"])
            forward_matrix[i, j] = bool(row["forward_fill_exposed"])
            linear_matrix[i, j] = bool(row["linear_interpolation_exposed"])

    rows: list[dict] = []
    for split_name, start, stop in (
        ("train", CAMERA_LAGS, split.train_end),
        ("validation", split.train_end, split.val_end),
        ("test", split.val_end, len(dates)),
    ):
        target_indices = list(range(start, stop))
        for asset_index, asset in [*(enumerate(assets)), (None, "ALL")]:
            windows_any: list[bool] = []
            windows_forward: list[bool] = []
            windows_linear: list[bool] = []
            exposed_cells = forward_cells = linear_cells = total_cells = 0
            target_exposed = 0
            for target_index in target_indices:
                if asset_index is None:
                    window_any = any_matrix[target_index - CAMERA_LAGS : target_index, :]
                    window_forward = forward_matrix[target_index - CAMERA_LAGS : target_index, :]
                    window_linear = linear_matrix[target_index - CAMERA_LAGS : target_index, :]
                    target_flags = any_matrix[target_index, :]
                else:
                    window_any = any_matrix[target_index - CAMERA_LAGS : target_index, asset_index]
                    window_forward = forward_matrix[target_index - CAMERA_LAGS : target_index, asset_index]
                    window_linear = linear_matrix[target_index - CAMERA_LAGS : target_index, asset_index]
                    target_flags = np.asarray([any_matrix[target_index, asset_index]])
                windows_any.append(bool(window_any.any()))
                windows_forward.append(bool(window_forward.any()))
                windows_linear.append(bool(window_linear.any()))
                exposed_cells += int(window_any.sum())
                forward_cells += int(window_forward.sum())
                linear_cells += int(window_linear.sum())
                total_cells += int(window_any.size)
                target_exposed += int(target_flags.any())

            origins = len(target_indices)
            rows.append(
                {
                    "split": split_name,
                    "asset": asset,
                    "forecast_origins": origins,
                    "input_return_cells": total_cells,
                    "exposed_input_return_cells": exposed_cells,
                    "exposed_input_return_cells_pct": _percentage(exposed_cells, total_cells),
                    "windows_with_any_exposed_input": int(sum(windows_any)),
                    "windows_with_any_exposed_input_pct": _percentage(sum(windows_any), origins),
                    "windows_with_forward_fill_exposure": int(sum(windows_forward)),
                    "windows_with_linear_interpolation_exposure": int(sum(windows_linear)),
                    "targets_with_exposed_return": target_exposed,
                }
            )
    return pd.DataFrame(rows)


def build_interpolation_dates(
    provenance_long: pd.DataFrame,
    price_panel: pd.DataFrame,
    returns: pd.DataFrame,
    split: object,
) -> pd.DataFrame:
    interpolation = provenance_long.loc[
        provenance_long["provenance"].eq(LINEAR_INTERPOLATED)
    ].copy()
    price_dates = list(pd.to_datetime(price_panel["date"]))
    price_positions = {date: i for i, date in enumerate(price_dates)}
    return_dates = list(pd.to_datetime(returns["date"]))
    return_positions = {date: i for i, date in enumerate(return_dates)}

    records: list[dict] = []
    for row in interpolation.itertuples(index=False):
        price_date = pd.Timestamp(row.date)
        endpoint_dates: list[pd.Timestamp] = []
        if bool(row.row_kept):
            price_index = price_positions[price_date]
            for index in (price_index, price_index + 1):
                if index < len(price_dates) and price_dates[index] in return_positions:
                    endpoint_dates.append(price_dates[index])

        impacted_targets: set[int] = set()
        for endpoint_date in endpoint_dates:
            return_index = return_positions[endpoint_date]
            lower = max(CAMERA_LAGS, return_index + 1)
            upper = min(len(return_dates) - 1, return_index + CAMERA_LAGS)
            impacted_targets.update(range(lower, upper + 1))

        split_counts = {"train": 0, "validation": 0, "test": 0}
        for target_index in impacted_targets:
            if target_index < split.train_end:
                split_counts["train"] += 1
            elif target_index < split.val_end:
                split_counts["validation"] += 1
            else:
                split_counts["test"] += 1
        records.append(
            {
                "date": price_date,
                "asset": row.asset,
                "interpolated_price": row.price_after_policy,
                "row_kept": bool(row.row_kept),
                "affected_return_endpoints": ";".join(
                    date.strftime("%Y-%m-%d") for date in endpoint_dates
                ),
                "affected_supervised_windows": len(impacted_targets),
                "affected_train_windows": split_counts["train"],
                "affected_validation_windows": split_counts["validation"],
                "affected_test_windows": split_counts["test"],
            }
        )
    return pd.DataFrame(records)


def render_report(
    provenance: pd.DataFrame,
    interpolation_dates: pd.DataFrame,
    return_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
    max_abs_difference: float,
    raw_hash_count: int,
) -> str:
    kept = provenance.loc[provenance["row_kept"]]
    provenance_counts = (
        provenance.groupby(["asset", "provenance"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    sse_return = return_summary.loc[return_summary["asset"].eq("sse")]
    sse_window = window_summary.loc[window_summary["asset"].eq("sse")]
    all_window = window_summary.loc[window_summary["asset"].eq("ALL")]
    return "\n".join(
        [
            "# Interpolation Exposure Audit",
            "",
            "## Outcome",
            "",
            "The camera-ready VN-Index-anchored price panel was reconstructed with cell-level provenance. "
            "The reconstruction matches the saved canonical panel; this audit did not alter preprocessing, "
            "raw data, model outputs or the manuscript.",
            "",
            "## Reproduction checks",
            "",
            f"- Raw CSV files hashed before and after the audit: {raw_hash_count}; all unchanged.",
            f"- Maximum absolute price difference from the saved canonical panel: `{max_abs_difference:.3g}`.",
            f"- Anchored price cells audited: {len(provenance):,}; cells retained after complete-case filtering: {len(kept):,}.",
            f"- Linearly interpolated cells: {len(interpolation_dates):,}; all belong to SSE: "
            f"`{bool(interpolation_dates.empty or interpolation_dates['asset'].eq('sse').all())}`.",
            "",
            "## Price-cell provenance",
            "",
            provenance_counts.to_markdown(index=False),
            "",
            "## SSE return-endpoint exposure by target-date split",
            "",
            sse_return.to_markdown(index=False),
            "",
            "## SSE supervised-window exposure",
            "",
            sse_window.to_markdown(index=False),
            "",
            "## Any-asset supervised-window exposure",
            "",
            all_window.to_markdown(index=False),
            "",
            "## Interpretation boundary",
            "",
            "- Forward-filled or interpolated inputs are reported as exposure, not as evidence of causal bias or harmlessness.",
            "- The main-calendar results, common-day robustness results and extra timing-shift results were not changed.",
            "- A supervised window is marked exposed when at least one lagged return uses a filled or interpolated price endpoint.",
            "- Linear interpolation is specific to SSE under the frozen camera-ready preprocessing contract.",
        ]
    )


def run_audit(output_dir: Path) -> dict[str, object]:
    raw_before = raw_csv_hashes()
    price_panel, provenance, _ = reconstruct_main_panel_with_provenance()
    max_abs_difference = assert_matches_canonical(price_panel)
    returns, _ = safe_log_return_panel(price_panel)
    return_detail, split = build_return_exposure_detail(price_panel, provenance, returns)
    return_summary = summarize_return_exposure(return_detail)
    window_summary = summarize_window_exposure(returns, return_detail, split)
    interpolation_dates = build_interpolation_dates(provenance, price_panel, returns, split)
    raw_after = raw_csv_hashes()
    if raw_before != raw_after:
        raise AssertionError("At least one raw CSV changed during the interpolation audit")

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance.to_csv(output_dir / "price_cell_provenance.csv", index=False)
    interpolation_dates.to_csv(output_dir / "interpolation_dates.csv", index=False)
    return_summary.to_csv(output_dir / "return_exposure_summary.csv", index=False)
    window_summary.to_csv(output_dir / "window_exposure_summary.csv", index=False)
    report = render_report(
        provenance,
        interpolation_dates,
        return_summary,
        window_summary,
        max_abs_difference,
        len(raw_before),
    )
    (output_dir / "interpolation_audit.md").write_text(report, encoding="utf-8")
    return {
        "raw_hashes": raw_before,
        "max_abs_difference": max_abs_difference,
        "price_cells": len(provenance),
        "interpolated_cells": len(interpolation_dates),
        "return_summary": return_summary,
        "window_summary": window_summary,
    }
