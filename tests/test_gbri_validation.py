from __future__ import annotations

import unittest

import numpy as np

from src.gbri_validation import (
    EVENT_QUANTILE,
    HORIZONS,
    construct_forward_targets,
    load_and_validate_canonical_gbri,
)


class GbriValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.returns, cls.canonical, cls.split = load_and_validate_canonical_gbri()
        cls.targets, cls.thresholds = construct_forward_targets(cls.returns, cls.canonical)

    def test_reconstructed_test_gbri_matches_canonical_series(self):
        test = self.canonical.loc[self.canonical["segment"].eq("test")]
        self.assertEqual(len(test), 425)
        self.assertTrue(np.isfinite(test["gbri"]).all())

    def test_forward_targets_use_strictly_later_returns(self):
        for horizon in HORIZONS:
            valid = self.targets[f"forward_rv_{horizon}d"].notna()
            self.assertTrue(
                (
                    self.targets.loc[valid, f"forward_{horizon}d_start_date"]
                    > self.targets.loc[valid, "date"]
                ).all()
            )
            first = self.targets.index[valid][0]
            expected = np.sqrt(
                np.mean(
                    np.square(
                        self.returns["return_vnindex"].iloc[first + 1 : first + horizon + 1]
                    )
                )
            )
            self.assertAlmostEqual(self.targets.loc[first, f"forward_rv_{horizon}d"], expected)

    def test_event_threshold_uses_training_targets_only(self):
        for horizon in HORIZONS:
            column = f"forward_rv_{horizon}d"
            expected = float(
                self.targets.loc[self.targets["segment"].eq("train"), column]
                .dropna()
                .quantile(EVENT_QUANTILE)
            )
            self.assertAlmostEqual(self.thresholds[horizon], expected)
            test_only = float(
                self.targets.loc[self.targets["segment"].eq("test"), column]
                .dropna()
                .quantile(EVENT_QUANTILE)
            )
            self.assertNotAlmostEqual(self.thresholds[horizon], test_only)


if __name__ == "__main__":
    unittest.main()
