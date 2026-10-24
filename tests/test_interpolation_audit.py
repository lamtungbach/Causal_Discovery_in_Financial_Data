from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from config import ASSETS, CAMERA_READY_DIR
from src.camera_ready_data import build_price_panel
from src.interpolation_audit import (
    FORWARD_FILLED,
    LINEAR_INTERPOLATED,
    apply_main_fill_with_provenance,
    raw_csv_hashes,
    reconstruct_main_panel_with_provenance,
)


class InterpolationAuditTests(unittest.TestCase):
    def test_synthetic_gap_distinguishes_forward_fill_and_interpolation(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        panel = pd.DataFrame({"date": dates})
        for meta in ASSETS.values():
            panel[meta["price"]] = np.arange(10.0, 16.0)

        oil_col = ASSETS["oil"]["price"]
        sse_col = ASSETS["sse"]["price"]
        panel.loc[1, oil_col] = np.nan
        panel[sse_col] = [10.0, np.nan, np.nan, np.nan, np.nan, 20.0]

        filled, provenance = apply_main_fill_with_provenance(panel)

        self.assertEqual(provenance.loc[1, oil_col], FORWARD_FILLED)
        self.assertEqual(provenance.loc[1, sse_col], FORWARD_FILLED)
        self.assertEqual(provenance.loc[4, sse_col], LINEAR_INTERPOLATED)
        self.assertEqual(filled.loc[4, sse_col], 15.0)

    def test_reconstruction_matches_current_and_saved_canonical_panel(self):
        reconstructed, _, _ = reconstruct_main_panel_with_provenance()
        current, _ = build_price_panel("main")
        pd.testing.assert_frame_equal(reconstructed, current, check_exact=True)

        saved = pd.read_csv(
            CAMERA_READY_DIR / "data" / "main_price_panel.csv", parse_dates=["date"]
        )
        self.assertEqual(reconstructed.columns.tolist(), saved.columns.tolist())
        self.assertTrue(reconstructed["date"].equals(saved["date"]))
        np.testing.assert_allclose(
            reconstructed.drop(columns="date").to_numpy(dtype=float),
            saved.drop(columns="date").to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_reconstruction_does_not_modify_raw_csvs(self):
        before = raw_csv_hashes()
        reconstruct_main_panel_with_provenance()
        after = raw_csv_hashes()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
