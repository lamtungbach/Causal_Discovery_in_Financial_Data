"""
tests/test_metrics.py
Unit tests cho evaluation metrics.

Chạy:
    python -m pytest tests/test_metrics.py -v
"""

import sys
import pytest
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    structural_hamming_distance,
    compute_metrics,
    MetricResult,
)
from src.evaluation.bootstrap import bootstrap_ci, BootstrapResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_dag():
    """pred = true → SHD=0, P=1, R=1, F1=1."""
    true = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ])
    return true, true.copy()


@pytest.fixture
def empty_pred():
    """pred = zeros → miss tất cả cạnh."""
    true = np.array([
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    pred = np.zeros((5, 5), dtype=int)
    return true, pred


@pytest.fixture
def full_pred():
    """pred = ones (no diagonal)."""
    true = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    pred = np.ones((5, 5), dtype=int)
    np.fill_diagonal(pred, 0)
    return true, pred


@pytest.fixture
def reversed_dag():
    """pred đảo ngược true."""
    true = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    pred = np.array([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    return true, pred


@pytest.fixture
def realistic_dag():
    """DAG thực tế: pred đúng một phần."""
    true = np.array([
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ])
    pred = np.array([
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
    ])
    return true, pred


# ─────────────────────────────────────────────────────────────────────────────
# Tests: structural_hamming_distance
# ─────────────────────────────────────────────────────────────────────────────

class TestStructuralHammingDistance:

    def test_identical_dag_shd_zero(self, perfect_dag):
        true, pred = perfect_dag
        assert structural_hamming_distance(true, pred) == 0

    def test_empty_pred_shd_equals_true_edges(self, empty_pred):
        true, pred = empty_pred
        shd = structural_hamming_distance(true, pred)
        assert shd == int(true.sum())

    def test_reversed_edges_counted_once(self, reversed_dag):
        """Cạnh đảo ngược chỉ tính 1 lần, không phải 2."""
        true, pred = reversed_dag
        assert structural_hamming_distance(true, pred) == 2

    def test_shd_non_negative(self, realistic_dag):
        true, pred = realistic_dag
        assert structural_hamming_distance(true, pred) >= 0

    def test_shd_symmetric_fp_fn(self):
        true = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        pred = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        assert structural_hamming_distance(true, pred) == 2

    def test_extra_edges_add_to_shd(self):
        true = np.zeros((4, 4), dtype=int)
        pred = np.zeros((4, 4), dtype=int)
        pred[0, 1] = pred[2, 3] = 1
        assert structural_hamming_distance(true, pred) == 2

    def test_all_zeros_both_zero(self):
        assert structural_hamming_distance(
            np.zeros((5, 5), dtype=int),
            np.zeros((5, 5), dtype=int)
        ) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMetrics:

    def test_perfect_prediction(self, perfect_dag):
        true, pred = perfect_dag
        m = compute_metrics(true, pred)
        assert m.precision == pytest.approx(1.0, abs=1e-6)
        assert m.recall    == pytest.approx(1.0, abs=1e-6)
        assert m.f1        == pytest.approx(1.0, abs=1e-6)
        assert m.shd       == 0

    def test_empty_pred_zero_precision_zero_recall(self, empty_pred):
        true, pred = empty_pred
        m = compute_metrics(true, pred)
        assert m.precision == pytest.approx(0.0, abs=1e-6)
        assert m.recall    == pytest.approx(0.0, abs=1e-6)
        assert m.f1        == pytest.approx(0.0, abs=1e-6)

    def test_full_pred_recall_is_one(self, full_pred):
        """pred=ones → Recall=1 (tìm được tất cả cạnh thật)."""
        true, pred = full_pred
        m = compute_metrics(true, pred)
        assert m.recall == pytest.approx(1.0, abs=1e-6)

    def test_full_pred_low_precision(self, full_pred):
        """pred=ones → Precision rất thấp."""
        true, pred = full_pred
        m = compute_metrics(true, pred)
        assert m.precision < 0.5

    def test_realistic_dag_partial(self, realistic_dag):
        true, pred = realistic_dag
        m = compute_metrics(true, pred)
        assert 0.0 < m.precision <= 1.0
        assert 0.0 < m.recall    <= 1.0
        assert 0.0 < m.f1        <= 1.0

    def test_f1_harmonic_mean(self, realistic_dag):
        true, pred = realistic_dag
        m = compute_metrics(true, pred)
        if m.precision + m.recall > 0:
            expected = 2 * m.precision * m.recall / (m.precision + m.recall)
            assert m.f1 == pytest.approx(expected, abs=1e-6)

    def test_tp_fp_fn_tn_sum_correct(self, realistic_dag):
        """TP + FP + FN + TN = d*(d-1) — không tính diagonal."""
        true, pred = realistic_dag
        m = compute_metrics(true, pred)
        d = true.shape[0]
        assert m.tp + m.fp + m.fn + m.tn == d * (d - 1)

    def test_metric_result_is_dataclass(self, perfect_dag):
        """compute_metrics trả về MetricResult."""
        true, pred = perfect_dag
        m = compute_metrics(true, pred)
        assert isinstance(m, MetricResult)

    def test_to_dict_has_all_keys(self, realistic_dag):
        true, pred = realistic_dag
        d = compute_metrics(true, pred).to_dict()
        for k in ["SHD", "Precision", "Recall (TPR)", "F1-score", "FPR"]:
            assert k in d, f"Thiếu key '{k}'"

    def test_fpr_range(self, realistic_dag):
        true, pred = realistic_dag
        m = compute_metrics(true, pred)
        assert 0.0 <= m.fpr <= 1.0

    @pytest.mark.parametrize("d", [3, 4, 5, 7])
    def test_different_sizes(self, d):
        rng  = np.random.RandomState(42)
        true = (rng.rand(d, d) > 0.7).astype(int)
        pred = (rng.rand(d, d) > 0.7).astype(int)
        np.fill_diagonal(true, 0)
        np.fill_diagonal(pred, 0)
        m = compute_metrics(true, pred)
        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.recall    <= 1.0
        assert 0.0 <= m.f1        <= 1.0
        assert m.shd >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: bootstrap_ci
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapCI:

    def test_returns_bootstrap_result(self, realistic_dag):
        true, pred = realistic_dag
        r = bootstrap_ci(true, pred, n_bootstrap=50, random_state=42)
        assert isinstance(r, BootstrapResult)

    def test_ci_lower_leq_base_leq_upper(self, realistic_dag):
        true, pred = realistic_dag
        r = bootstrap_ci(true, pred, n_bootstrap=200, random_state=42)
        assert r.prec_ci[0]   <= r.precision + 1e-9
        assert r.prec_ci[1]   >= r.precision - 1e-9
        assert r.recall_ci[0] <= r.recall    + 1e-9
        assert r.recall_ci[1] >= r.recall    - 1e-9
        assert r.f1_ci[0]     <= r.f1        + 1e-9
        assert r.f1_ci[1]     >= r.f1        - 1e-9

    def test_ci_width_positive(self, realistic_dag):
        true, pred = realistic_dag
        r = bootstrap_ci(true, pred, n_bootstrap=200, random_state=42)
        assert r.prec_ci[1]   - r.prec_ci[0]   >= 0
        assert r.recall_ci[1] - r.recall_ci[0] >= 0
        assert r.f1_ci[1]     - r.f1_ci[0]     >= 0

    def test_perfect_dag_perfect_metrics(self, perfect_dag):
        true, pred = perfect_dag
        r = bootstrap_ci(true, pred, n_bootstrap=100, random_state=0)
        assert r.precision == pytest.approx(1.0, abs=1e-6)
        assert r.recall    == pytest.approx(1.0, abs=1e-6)
        assert r.f1        == pytest.approx(1.0, abs=1e-6)

    def test_reproducible_with_same_seed(self, realistic_dag):
        true, pred = realistic_dag
        r1 = bootstrap_ci(true, pred, n_bootstrap=100, random_state=99)
        r2 = bootstrap_ci(true, pred, n_bootstrap=100, random_state=99)
        assert r1.f1       == r2.f1
        assert r1.f1_ci[0] == r2.f1_ci[0]
        assert r1.f1_ci[1] == r2.f1_ci[1]

    def test_different_seeds_same_base_metrics(self, realistic_dag):
        """Base metrics không stochastic — chỉ CI mới có thể khác."""
        true, pred = realistic_dag
        r1 = bootstrap_ci(true, pred, n_bootstrap=200, random_state=1)
        r2 = bootstrap_ci(true, pred, n_bootstrap=200, random_state=999)
        assert r1.precision == r2.precision
        assert r1.recall    == r2.recall

    def test_format_table_row_has_required_keys(self, realistic_dag):
        true, pred = realistic_dag
        r   = bootstrap_ci(true, pred, n_bootstrap=50, random_state=42)
        row = r.format_table_row("NOTEARS", runtime=12.34)
        for key in ["Method", "SHD ↓", "Precision ↑", "Recall ↑", "F1 ↑", "Runtime (s)"]:
            assert key in row, f"Thiếu key '{key}'"

    def test_to_dict_has_ci_keys(self, realistic_dag):
        true, pred = realistic_dag
        r = bootstrap_ci(true, pred, n_bootstrap=50, random_state=42)
        d = r.to_dict()
        for key in ["Precision CI 95%", "Recall CI 95%", "F1 CI 95%"]:
            assert key in d, f"Thiếu key '{key}'"

    @pytest.mark.parametrize("n_boot", [50, 100, 500])
    def test_various_bootstrap_sizes(self, realistic_dag, n_boot):
        true, pred = realistic_dag
        r = bootstrap_ci(true, pred, n_bootstrap=n_boot, random_state=42)
        assert r.n_bootstrap == n_boot
        assert 0.0 <= r.f1 <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_node_dag(self):
        """DAG 1 node → không có cạnh."""
        true = np.zeros((1, 1), dtype=int)
        pred = np.zeros((1, 1), dtype=int)
        assert structural_hamming_distance(true, pred) == 0
        m = compute_metrics(true, pred)
        assert m.shd == 0

    def test_fully_connected_true(self):
        """true=fully connected, pred=zeros → recall=0."""
        d    = 4
        true = np.ones((d, d), dtype=int)
        np.fill_diagonal(true, 0)
        pred = np.zeros((d, d), dtype=int)
        m    = compute_metrics(true, pred)
        assert m.recall    == pytest.approx(0.0, abs=1e-6)
        assert m.precision == pytest.approx(0.0, abs=1e-6)

    def test_diagonal_ignored(self):
        """Diagonal không được tính vào metrics."""
        true = np.eye(5, dtype=int)
        pred = np.eye(5, dtype=int)
        m    = compute_metrics(true, pred)
        assert m.tp == 0

    def test_shd_with_all_reversed(self):
        true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        pred = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        assert structural_hamming_distance(true, pred) == 2

    def test_large_random_dag_no_crash(self):
        """Không crash với d=10."""
        rng  = np.random.RandomState(123)
        d    = 10
        true = np.tril(rng.randint(0, 2, (d, d)), -1)
        pred = np.tril(rng.randint(0, 2, (d, d)), -1)
        shd  = structural_hamming_distance(true, pred)
        m    = compute_metrics(true, pred)
        assert shd >= 0
        assert 0 <= m.f1 <= 1


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])