from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from config import CAMERA_READY_DIR
from src.seed_uncertainty import (
    consensus_graph_table,
    edge_frequency_intervals,
    load_saved_seed_graphs,
    node_role_outputs,
    verify_against_canonical,
    wilson_interval,
)


class SeedUncertaintyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graphs = load_saved_seed_graphs()
        cls.edge_intervals = edge_frequency_intervals(cls.graphs)
        cls.role_intervals, cls.vni = node_role_outputs(cls.graphs)
        cls.consensus = consensus_graph_table(cls.edge_intervals)

    def test_reproduces_all_canonical_selection_frequencies(self):
        canonical = pd.read_csv(
            CAMERA_READY_DIR / "tables" / "edge_stability_20seeds.csv"
        )
        merged = canonical.merge(
            self.edge_intervals,
            on=["source", "target"],
            suffixes=("_canonical", "_reconstructed"),
            validate="one_to_one",
        )
        np.testing.assert_allclose(
            merged["selection_frequency_canonical"],
            merged["selection_frequency_reconstructed"],
            rtol=0.0,
            atol=0.0,
        )

    def test_consensus_080_contains_exactly_canonical_stable_edges(self):
        verify_against_canonical(self.edge_intervals, self.role_intervals, self.consensus)
        selected = self.consensus.loc[
            self.consensus["consensus_level"].eq(0.80)
            & self.consensus["selected_in_consensus"]
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual((selected.iloc[0]["source"], selected.iloc[0]["target"]), ("VNI", "GOLD"))

    def test_wilson_interval_is_bounded_and_contains_observed_proportion(self):
        lower, upper = wilson_interval(13, 20)
        self.assertTrue(0.0 <= lower <= 13 / 20 <= upper <= 1.0)


if __name__ == "__main__":
    unittest.main()
