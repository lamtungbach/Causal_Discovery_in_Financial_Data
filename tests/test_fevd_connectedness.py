from __future__ import annotations

import numpy as np
import pandas as pd

from src.fevd_connectedness import HORIZONS, fit_gfevd, generalized_fevd


def _stable_synthetic_var(seed: int = 73, observations: int = 450) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    transition = np.array(
        [
            [0.28, 0.08, 0.00],
            [0.02, 0.24, 0.06],
            [0.04, 0.00, 0.20],
        ]
    )
    covariance = np.array(
        [
            [1.00, 0.25, 0.10],
            [0.25, 0.80, 0.12],
            [0.10, 0.12, 0.65],
        ]
    )
    innovations = rng.multivariate_normal(np.zeros(3), covariance, size=observations)
    values = np.zeros((observations, 3))
    for index in range(1, observations):
        values[index] = transition @ values[index - 1] + innovations[index]
    return pd.DataFrame(values, columns=["A", "B", "C"])


def test_generalized_fevd_is_finite_nonnegative_and_row_normalized() -> None:
    coefs = np.array([[[0.3, 0.1], [0.0, 0.2]]])
    sigma = np.array([[1.0, 0.25], [0.25, 0.8]])
    matrix = generalized_fevd(coefs, sigma, horizon=10)
    assert np.isfinite(matrix).all()
    assert (matrix >= -1e-14).all()
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-12)


def test_fitted_gfevd_is_invariant_to_two_variable_orders() -> None:
    data = _stable_synthetic_var()
    _, reference = fit_gfevd(data, lags=2)
    for order in (["C", "B", "A"], ["B", "C", "A"]):
        _, permuted = fit_gfevd(data[order], lags=2)
        for horizon in HORIZONS:
            restored = permuted[horizon].loc[data.columns, data.columns]
            np.testing.assert_allclose(restored, reference[horizon], atol=1e-10, rtol=1e-10)


def test_both_prespecified_horizons_are_produced() -> None:
    data = _stable_synthetic_var()
    _, matrices = fit_gfevd(data, lags=2)
    assert set(matrices) == {10, 20}
