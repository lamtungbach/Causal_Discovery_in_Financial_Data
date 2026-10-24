from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from config import CAMERA_READY_DIR
from src.camera_ready_analysis import DISPLAY_LABELS
from src.graph_sensitivity import (
    QUANTILES,
    analyze_saved_matrices,
    load_all_saved_matrices,
    verify_monotonic_edge_counts,
    verify_q70_seed42,
)


class GraphSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrices = load_all_saved_matrices()
        (
            cls.graph_summary,
            cls.edge_frequency,
            cls.node_roles,
            cls.jaccard,
            cls.diagnostics,
        ) = analyze_saved_matrices(cls.matrices)

    def test_q70_seed42_reproduces_canonical_graph_and_threshold(self):
        verify_q70_seed42(self.diagnostics)
        reconstructed = self.diagnostics["retained"][(42, 0.70)]
        canonical = pd.read_csv(
            CAMERA_READY_DIR / "tables" / "ablation_importance_retained_seed42.csv"
        ).set_index("source").loc[DISPLAY_LABELS, DISPLAY_LABELS].to_numpy(float)
        self.assertEqual(int(np.count_nonzero(reconstructed)), 7)
        np.testing.assert_allclose(reconstructed, canonical, rtol=0.0, atol=1e-15)

    def test_higher_quantiles_never_retain_more_edges(self):
        verify_monotonic_edge_counts(self.graph_summary)
        for _, frame in self.graph_summary.groupby("seed"):
            counts = frame.set_index("quantile").loc[list(QUANTILES), "edge_count"].to_numpy()
            self.assertTrue(np.all(np.diff(counts) <= 0))


if __name__ == "__main__":
    unittest.main()
