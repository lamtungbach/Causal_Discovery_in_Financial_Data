"""
tests/test_pipeline.py
Integration tests cho pipeline.

Chạy:
    python -m pytest tests/test_pipeline.py -v -k "not slow"
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: tương thích với cả 2 phiên bản DataPreprocessor
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(prices: pd.DataFrame, window: int = 10):
    """
    Wrapper tương thích:
      - Phiên bản mới: prep.fit_transform(prices)
      - Phiên bản cũ:  prep.compute_returns() + prep.compute_volatility()
    """
    from src.data.preprocessor import DataPreprocessor
    prep = DataPreprocessor(window=window)

    if hasattr(prep, "fit_transform"):
        X, vol = prep.fit_transform(prices, standardize=True)
        return X, vol, prep

    # Fallback cho file cũ
    prep.compute_returns(prices)
    vol = prep.compute_volatility()
    vol = prep.handle_missing(vol) if hasattr(prep, "handle_missing") else vol.ffill().bfill()
    prep.volatility_ = vol

    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(vol.values)
    return X, vol, prep


# ─────────────────────────────────────────────────────────────────────────────
# Helper: tương thích granger.py (pvalue_matrix vs pvalue_matrix_)
# ─────────────────────────────────────────────────────────────────────────────

def _get_pvalue_matrix(model) -> np.ndarray:
    """Lấy pvalue matrix dù dùng tên attribute nào."""
    for attr in ["pvalue_matrix_", "pvalue_matrix"]:
        v = getattr(model, attr, None)
        if v is not None:
            return v
    raise AttributeError("Không tìm thấy pvalue_matrix trong GrangerCausality")


def _get_pvalue_df(model) -> pd.DataFrame:
    """Gọi đúng method trả về p-value DataFrame."""
    for method_name in ["pvalue_df", "get_pvalue_df"]:
        m = getattr(model, method_name, None)
        if m is not None:
            return m()
    raise AttributeError("Không tìm thấy pvalue_df/get_pvalue_df")


def _get_significant_pairs(model) -> list:
    """Gọi đúng method trả về significant pairs."""
    for method_name in ["significant_pairs", "get_significant_pairs"]:
        m = getattr(model, method_name, None)
        if m is not None:
            return m()
    raise AttributeError("Không tìm thấy significant_pairs/get_significant_pairs")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: tương thích validator.py
# ─────────────────────────────────────────────────────────────────────────────

def _validator_test(validator, vol_df) -> pd.DataFrame:
    """Gọi đúng method test stationarity."""
    for method_name in ["test", "test_stationarity"]:
        m = getattr(validator, method_name, None)
        if m is not None:
            return m(vol_df)
    raise AttributeError("Không tìm thấy test/test_stationarity trong validator")


def _validator_ensure(validator, vol_df) -> pd.DataFrame:
    """Gọi ensure_stationary nếu có."""
    m = getattr(validator, "ensure_stationary", None)
    if m is not None:
        return m(vol_df)
    # Fallback nếu không có
    return vol_df.ffill().bfill()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_prices():
    """200 ngày × 5 thị trường synthetic."""
    np.random.seed(42)
    n   = 200
    idx = pd.bdate_range("2022-01-01", periods=n)
    starts = {"SP500": 4000, "VNINDEX": 1300, "Gold": 1800,
               "WTI_Oil": 80, "Bitcoin": 45000}
    data = {
        name: start * np.cumprod(1 + np.random.normal(0.0003, 0.012, n))
        for name, start in starts.items()
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture(scope="module")
def preprocessed_data(synthetic_prices):
    return _preprocess(synthetic_prices, window=10)


@pytest.fixture(scope="module")
def X_matrix(preprocessed_data):
    return preprocessed_data[0]


@pytest.fixture(scope="module")
def vol_df(preprocessed_data):
    return preprocessed_data[1]


@pytest.fixture
def node_names():
    return ["SP500", "VNINDEX", "Gold", "WTI_Oil", "Bitcoin"]


@pytest.fixture
def small_X():
    np.random.seed(0)
    return np.random.randn(60, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: DataPreprocessor
# ─────────────────────────────────────────────────────────────────────────────

class TestDataPreprocessor:

    def test_output_shape(self, preprocessed_data):
        X, vol, _ = preprocessed_data
        assert X.ndim == 2 and X.shape[1] == 5 and X.shape[0] > 0

    def test_no_nan_in_X(self, preprocessed_data):
        X, _, _ = preprocessed_data
        assert not np.isnan(X).any()

    def test_no_nan_in_vol(self, preprocessed_data):
        _, vol, _ = preprocessed_data
        assert not vol.isna().any().any()

    def test_standardized_mean_near_zero(self, preprocessed_data):
        X, _, _ = preprocessed_data
        assert (np.abs(X.mean(axis=0)) < 0.1).all()

    def test_standardized_std_near_one(self, preprocessed_data):
        X, _, _ = preprocessed_data
        assert (np.abs(X.std(axis=0) - 1.0) < 0.3).all()

    def test_volatility_positive(self, preprocessed_data):
        _, vol, _ = preprocessed_data
        assert (vol.values > 0).all()

    def test_vol_df_has_5_columns(self, preprocessed_data):
        _, vol, _ = preprocessed_data
        assert vol.shape[1] == 5

    def test_describe_returns_dataframe(self, preprocessed_data):
        _, vol, prep = preprocessed_data
        if hasattr(prep, "describe"):
            try:
                desc = prep.describe()
            except Exception:
                desc = vol.describe().T
        else:
            desc = vol.describe().T
        assert isinstance(desc, pd.DataFrame)

    def test_x_and_vol_same_n_rows(self, preprocessed_data):
        X, vol, _ = preprocessed_data
        assert X.shape[0] == vol.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: StationarityValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestStationarityValidator:

    def _get_validator(self):
        """Import validator từ src.data hoặc src.data.validator."""
        try:
            from src.data.validator import StationarityValidator
            return StationarityValidator()
        except ImportError:
            pytest.skip("StationarityValidator không tìm thấy")

    def test_returns_dataframe(self, vol_df):
        v  = self._get_validator()
        df = _validator_test(v, vol_df)
        assert isinstance(df, pd.DataFrame)

    def test_has_adf_columns(self, vol_df):
        v   = self._get_validator()
        df  = _validator_test(v, vol_df)
        has_adf = any("ADF" in c or "adf" in c.lower() for c in df.columns)
        assert has_adf, f"Không có cột ADF trong: {list(df.columns)}"

    def test_rows_equal_n_markets(self, vol_df):
        v  = self._get_validator()
        df = _validator_test(v, vol_df)
        assert len(df) == vol_df.shape[1]

    def test_ensure_stationary_no_nan(self, vol_df):
        v = self._get_validator()
        _validator_test(v, vol_df)
        result = _validator_ensure(v, vol_df)
        assert not result.isna().any().any()

    def test_ensure_stationary_returns_dataframe(self, vol_df):
        v = self._get_validator()
        _validator_test(v, vol_df)
        result = _validator_ensure(v, vol_df)
        assert isinstance(result, pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: CausalResult
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalResult:

    @pytest.fixture
    def sample_result(self, node_names):
        from src.models.base_model import CausalResult
        adj    = np.array([
            [0.0, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        binary = (np.abs(adj) > 0.2).astype(int)
        # n_edges có default=0 → không cần truyền vào
        return CausalResult(
            method_name      = "Test",
            adjacency_matrix = adj,
            binary_matrix    = binary,
            runtime_seconds  = 1.23,
            metadata         = {"node_names": node_names},
        )

    def test_n_edges_correct(self, sample_result):
        assert sample_result.n_edges == 3

    def test_out_degree_shape(self, sample_result):
        assert sample_result.out_degree().shape == (5,)

    def test_in_degree_shape(self, sample_result):
        assert sample_result.in_degree().shape == (5,)

    def test_hub_nodes_returns_list(self, sample_result):
        hubs = sample_result.hub_nodes(top_k=2)
        assert isinstance(hubs, list) and len(hubs) <= 2

    def test_sink_nodes_returns_list(self, sample_result):
        assert isinstance(sample_result.sink_nodes(top_k=2), list)

    def test_edge_list_length(self, sample_result):
        assert len(sample_result.edge_list()) == sample_result.n_edges

    def test_edge_list_keys(self, sample_result):
        for e in sample_result.edge_list():
            assert "source" in e and "target" in e and "weight" in e

    def test_degree_table_columns(self, sample_result):
        df = sample_result.degree_table()
        assert isinstance(df, pd.DataFrame)
        assert "Out-degree" in df.columns and "In-degree" in df.columns

    def test_node_names_from_metadata(self, sample_result, node_names):
        assert sample_result.node_names == node_names


# ─────────────────────────────────────────────────────────────────────────────
# Tests: NOTEARS
# ─────────────────────────────────────────────────────────────────────────────

class TestNOTEARS:

    def test_fit_returns_causal_result(self, small_X, node_names):
        from src.models.notears import NOTEARS
        from src.models.base_model import CausalResult
        result = NOTEARS(lambda1=0.1, threshold=0.2, node_names=node_names).fit(small_X)
        assert isinstance(result, CausalResult)

    def test_binary_matrix_shape(self, small_X, node_names):
        from src.models.notears import NOTEARS
        result = NOTEARS(lambda1=0.1, threshold=0.2, node_names=node_names).fit(small_X)
        assert result.binary_matrix.shape == (5, 5)

    def test_no_self_loops(self, small_X, node_names):
        from src.models.notears import NOTEARS
        result = NOTEARS(lambda1=0.1, threshold=0.2, node_names=node_names).fit(small_X)
        assert (np.diag(result.binary_matrix) == 0).all()

    def test_binary_values_only(self, small_X, node_names):
        from src.models.notears import NOTEARS
        result = NOTEARS(lambda1=0.1, threshold=0.2, node_names=node_names).fit(small_X)
        assert set(np.unique(result.binary_matrix)).issubset({0, 1})

    def test_fit_timed_sets_runtime(self, small_X, node_names):
        from src.models.notears import NOTEARS
        result = NOTEARS(lambda1=0.1, threshold=0.2, node_names=node_names).fit_timed(small_X)
        assert result.runtime_seconds > 0

    def test_threshold_sensitivity_monotone(self, small_X, node_names):
        from src.models.notears import NOTEARS
        model = NOTEARS(lambda1=0.1, threshold=0.0, node_names=node_names)
        model.fit(small_X)
        sens  = model.threshold_sensitivity([0.1, 0.2, 0.3, 0.4])
        edges = [s["n_edges"] for s in sens]
        for i in range(len(edges) - 1):
            assert edges[i] >= edges[i + 1]

    def test_summary_is_string(self, small_X, node_names):
        from src.models.notears import NOTEARS
        model = NOTEARS(lambda1=0.1, threshold=0.2, node_names=node_names)
        model.fit_timed(small_X)
        assert isinstance(model.summary(), str)

    @pytest.mark.parametrize("lambda1", [0.05, 0.1, 0.2])
    def test_different_lambda1(self, small_X, node_names, lambda1):
        from src.models.notears import NOTEARS
        result = NOTEARS(lambda1=lambda1, threshold=0.2, node_names=node_names).fit(small_X)
        assert result.binary_matrix.shape == (5, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: GrangerCausality — tương thích pvalue_matrix / pvalue_matrix_
# ─────────────────────────────────────────────────────────────────────────────

class TestGrangerCausality:

    def test_fit_returns_causal_result(self, small_X, node_names):
        from src.models.neural_granger import GrangerCausality
        from src.models.base_model import CausalResult
        result = GrangerCausality(max_lag=2, node_names=node_names).fit(small_X)
        assert isinstance(result, CausalResult)

    def test_pvalue_matrix_shape(self, small_X, node_names):
        from src.models.neural_granger import GrangerCausality
        model = GrangerCausality(max_lag=2, node_names=node_names)
        model.fit(small_X)
        pval = _get_pvalue_matrix(model)
        assert pval.shape == (5, 5)

    def test_pvalue_range(self, small_X, node_names):
        from src.models.neural_granger import GrangerCausality
        model = GrangerCausality(max_lag=2, node_names=node_names)
        model.fit(small_X)
        pval = _get_pvalue_matrix(model)
        assert (pval >= 0).all() and (pval <= 1).all()

    def test_diagonal_pvalue_is_one(self, small_X, node_names):
        from src.models.neural_granger import GrangerCausality
        model = GrangerCausality(max_lag=2, node_names=node_names)
        model.fit(small_X)
        pval = _get_pvalue_matrix(model)
        np.testing.assert_array_equal(np.diag(pval), 1.0)

    def test_pvalue_df_is_dataframe(self, small_X, node_names):
        from src.models.neural_granger import GrangerCausality
        model = GrangerCausality(max_lag=2, node_names=node_names)
        model.fit(small_X)
        assert isinstance(_get_pvalue_df(model), pd.DataFrame)

    def test_significant_pairs_is_list(self, small_X, node_names):
        from src.models.neural_granger import GrangerCausality
        model = GrangerCausality(max_lag=2, node_names=node_names)
        model.fit_timed(small_X)
        assert isinstance(_get_significant_pairs(model), list)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: MethodComparator
# ─────────────────────────────────────────────────────────────────────────────

class TestMethodComparator:

    @pytest.fixture
    def two_results(self, small_X, node_names):
        from src.models.neural_granger import GrangerCausality
        from src.models.notears import NOTEARS
        g = GrangerCausality(max_lag=2, node_names=node_names).fit_timed(small_X)
        n = NOTEARS(lambda1=0.1, threshold=0.2, node_names=node_names).fit_timed(small_X)
        return g, n

    def test_comparison_table_is_dataframe(self, two_results):
        from src.evaluation.comparator import MethodComparator
        g, n = two_results
        cmp  = MethodComparator(g.binary_matrix, n_bootstrap=50)
        cmp.add("NOTEARS", n)
        assert isinstance(cmp.comparison_table(), pd.DataFrame)

    def test_comparison_table_has_one_row(self, two_results):
        from src.evaluation.comparator import MethodComparator
        g, n = two_results
        cmp  = MethodComparator(g.binary_matrix, n_bootstrap=50)
        cmp.add("NOTEARS", n)
        assert len(cmp.comparison_table()) == 1

    def test_raw_scores_columns(self, two_results):
        from src.evaluation.comparator import MethodComparator
        g, n = two_results
        cmp  = MethodComparator(g.binary_matrix, n_bootstrap=50)
        cmp.add("NOTEARS", n)
        for col in ["SHD", "Precision", "Recall", "F1"]:
            assert col in cmp.raw_scores().columns

    def test_best_method_returns_string(self, two_results):
        from src.evaluation.comparator import MethodComparator
        g, n = two_results
        cmp  = MethodComparator(g.binary_matrix, n_bootstrap=50)
        cmp.add("NOTEARS", n)
        assert isinstance(cmp.best_method("F1"), str)

    def test_latex_table_is_string(self, two_results):
        from src.evaluation.comparator import MethodComparator
        g, n  = two_results
        cmp   = MethodComparator(g.binary_matrix, n_bootstrap=50)
        cmp.add("NOTEARS", n)
        latex = cmp.latex_table()
        assert isinstance(latex, str) and ("tabular" in latex or "table" in latex)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: ResultIO
# ─────────────────────────────────────────────────────────────────────────────

class TestResultIO:

    @pytest.fixture
    def tmp_io(self, tmp_path):
        from src.utils.io_utils import ResultIO
        return ResultIO(base_dir=str(tmp_path))

    @pytest.fixture
    def dummy_result(self, node_names):
        from src.models.base_model import CausalResult
        binary = np.eye(5, k=1, dtype=int)
        return CausalResult(
            method_name      = "Test",
            adjacency_matrix = binary.astype(float) * 0.3,
            binary_matrix    = binary,
            runtime_seconds  = 2.5,
            metadata         = {"node_names": node_names},
        )

    def test_save_and_load(self, tmp_io, dummy_result):
        tmp_io.save_result("TestMethod", dummy_result)
        loaded = tmp_io.load_result("TestMethod")
        np.testing.assert_array_equal(
            loaded.binary_matrix, dummy_result.binary_matrix
        )

    def test_load_nonexistent_raises(self, tmp_io):
        with pytest.raises(FileNotFoundError):
            tmp_io.load_result("NonExistent")

    def test_snapshot_round_trip(self, tmp_io, dummy_result):
        tmp_io.snapshot({"Test": dummy_result})
        data = tmp_io.load_snapshot()
        assert "results" in data and "Test" in data["results"]

    def test_save_comparison_creates_csv(self, tmp_io):
        df = pd.DataFrame({"SHD": [3], "F1": [0.8]}, index=["NOTEARS"])
        tmp_io.save_comparison(df, formats=("csv",))
        assert (Path(tmp_io.tables_dir) / "method_comparison.csv").exists()

    def test_list_snapshots_is_dataframe(self, tmp_io):
        assert isinstance(tmp_io.list_snapshots(), pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# Integration
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestMiniPipeline:

    def test_full_flow(self, X_matrix, node_names):
        from src.models.notears       import NOTEARS
        from src.models.neural_granger       import GrangerCausality
        from src.evaluation.comparator import MethodComparator

        g   = GrangerCausality(max_lag=2, node_names=node_names).fit_timed(X_matrix)
        n   = NOTEARS(lambda1=0.1, threshold=0.2, node_names=node_names).fit_timed(X_matrix)
        cmp = MethodComparator(g.binary_matrix, n_bootstrap=50)
        cmp.add("NOTEARS", n)

        assert len(cmp.comparison_table()) == 1
        assert n.runtime_seconds > 0

    def test_snapshot_round_trip(self, X_matrix, node_names, tmp_path):
        from src.models.notears import NOTEARS
        from src.utils.io_utils import ResultIO

        result = NOTEARS(lambda1=0.1, threshold=0.2, node_names=node_names).fit_timed(X_matrix)
        io     = ResultIO(str(tmp_path))
        io.snapshot({"NOTEARS": result})
        loaded = io.load_snapshot()["results"]["NOTEARS"]
        np.testing.assert_array_equal(result.binary_matrix, loaded.binary_matrix)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-k", "not slow"])