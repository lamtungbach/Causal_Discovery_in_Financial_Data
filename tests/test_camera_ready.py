from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ASSETS, RETURN_COLS  # noqa: E402
from src.camera_ready_analysis import (  # noqa: E402
    exact_mcnemar,
    modified_diebold_mariano,
    pesaran_timmermann,
)
from src.camera_ready_data import prepare_supervised, safe_log_return_panel  # noqa: E402
from src.camera_ready_models import block_ablation_importance, sparsify_importance  # noqa: E402


class SourceZeroModel(nn.Module):
    def __init__(self, source: int):
        super().__init__()
        self.source = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, self.source].sum(dim=1)


class CameraReadyDataTests(unittest.TestCase):
    def test_non_positive_price_invalidates_two_adjacent_returns(self):
        dates = pd.date_range("2020-04-17", periods=4, freq="D")
        panel = pd.DataFrame({"date": dates})
        for asset, meta in ASSETS.items():
            panel[meta["price"]] = [10.0, 11.0, 12.0, 13.0]
        panel.loc[1, ASSETS["oil"]["price"]] = -37.63

        returns, invalid = safe_log_return_panel(panel)

        oil_invalid = invalid.loc[invalid["asset"].eq("oil")]
        self.assertEqual(
            oil_invalid["date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2020-04-18", "2020-04-19"],
        )
        self.assertEqual(returns["date"].dt.strftime("%Y-%m-%d").tolist(), ["2020-04-20"])

    def test_target_date_split_preserves_validation_and_test_context(self):
        n = 100
        rng = np.random.default_rng(5)
        frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n)})
        for column in RETURN_COLS:
            frame[column] = rng.normal(size=n)

        split = prepare_supervised(frame, lags=5)

        self.assertEqual(len(split.X_train), 65)
        self.assertEqual(len(split.X_val), 15)
        self.assertEqual(len(split.X_test), 15)
        self.assertEqual(split.X_train.shape[1:], (5, len(RETURN_COLS)))
        np.testing.assert_allclose(split.mean, frame[RETURN_COLS].iloc[:70].mean().to_numpy())

    def test_extra_timing_shift_operates_only_on_input_history(self):
        n = 100
        frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n)})
        for offset, column in enumerate(RETURN_COLS):
            frame[column] = np.arange(n, dtype=float) + offset
        normal = prepare_supervised(frame, lags=5)
        shifted = prepare_supervised(frame, lags=5, input_shift_sources=[RETURN_COLS[1]])
        self.assertEqual(len(normal.X_test), len(shifted.X_test))
        np.testing.assert_allclose(normal.Y_test, shifted.Y_test)
        np.testing.assert_allclose(
            shifted.X_test[:, 1:, 1], normal.X_test[:, :-1, 1], rtol=1e-6, atol=1e-6
        )


class CameraReadyModelTests(unittest.TestCase):
    def test_block_ablation_uses_unflattened_source_axis(self):
        X = np.ones((8, 3, 2), dtype=np.float32)
        models = [SourceZeroModel(1), SourceZeroModel(0)]
        Y = np.column_stack([np.full(8, 3.0), np.full(8, 3.0)]).astype(np.float32)
        importance, audit = block_ablation_importance(models, X, Y, ["A", "B"])
        self.assertGreater(importance[0, 1], 0)
        self.assertGreater(importance[1, 0], 0)
        self.assertEqual(len(audit), 2)

    def test_sparsification_preserves_only_upper_quantile(self):
        matrix = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0], [5.0, 6.0, 0.0]])
        retained, threshold = sparsify_importance(matrix, 0.5)
        self.assertEqual(threshold, 3.5)
        self.assertEqual(np.count_nonzero(retained), 3)
        self.assertTrue(np.all(np.diag(retained) == 0))


class CameraReadyInferenceTests(unittest.TestCase):
    def test_forecast_tests_return_finite_results(self):
        rng = np.random.default_rng(7)
        actual = rng.normal(size=200)
        ngc = actual + rng.normal(scale=0.5, size=200)
        lstm = actual + rng.normal(scale=0.8, size=200)
        dm = modified_diebold_mariano(actual, ngc, lstm)
        mc = exact_mcnemar(actual, ngc, lstm)
        pt = pesaran_timmermann(actual, ngc)
        self.assertTrue(np.isfinite(dm["statistic"]))
        self.assertTrue(0 <= dm["p_value"] <= 1)
        self.assertTrue(0 <= mc["p_value"] <= 1)
        self.assertTrue(np.isfinite(pt["statistic"]))
        self.assertTrue(0 <= pt["p_value"] <= 1)


if __name__ == "__main__":
    unittest.main()
