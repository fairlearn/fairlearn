# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier

from fairlearn.reductions import ErrorRate, ErrorRateResult
from fairlearn.reductions._moments.error_rate import _MESSAGE_BAD_COSTS

BAD_COSTS_EXAMPLES = [
    {"fp": 0.0, "fn": 0.0},
    {"fp": -1.0, "fn": 1.0},
    {"fp": 1.0, "fn": -1.0},
    {},
    {"FP": 1.0, "FN": 1.0},
    {"fp": 1.0},
    {"fn": 1.0},
]


@pytest.mark.parametrize("bad_costs", BAD_COSTS_EXAMPLES)
def test_bad_costs(bad_costs):
    with pytest.raises(ValueError) as execInfo:
        _ = ErrorRate(costs=bad_costs)
        assert _MESSAGE_BAD_COSTS in execInfo.value.args[0]


def _classification_data():
    rng = np.random.default_rng(42)
    X, y = make_classification(n_features=10, class_sep=0.1, random_state=42)
    X[:, -1] = rng.integers(0, 2, size=(X.shape[0],))
    sensitive_features = X[:, -1]
    return X, y, sensitive_features


def test_error_rate_numpy_and_pandas_compatible():
    """ErrorRate returns the same result for NumPy and pandas input."""
    X, y, sensitive_features = _classification_data()
    X_pd = pd.DataFrame(X)
    y_pd = pd.Series(y)
    sensitive_features_pd = pd.Series(sensitive_features)
    costs = {"fp": 0.1, "fn": 0.9}
    errorrate_np = ErrorRate(costs=costs)
    errorrate_pd = ErrorRate(costs=costs)
    errorrate_np.load_data(X, y, sensitive_features=sensitive_features)
    errorrate_pd.load_data(X_pd, y_pd, sensitive_features=sensitive_features_pd)

    classifier = HistGradientBoostingClassifier().fit(X, y)
    error_np = errorrate_np.gamma(classifier.predict)
    assert isinstance(error_np, ErrorRateResult)

    classifier = HistGradientBoostingClassifier().fit(X_pd, y_pd)
    error_pd = errorrate_pd.gamma(classifier.predict)
    assert isinstance(error_pd, ErrorRateResult)
    assert error_np == error_pd


def test_error_rate_uses_prediction_values_not_pandas_index():
    """Predictions are compared by position even when their index comes from X."""
    X = pd.DataFrame({"feature": [0, 1]}, index=[10, 20])
    error_rate = ErrorRate()
    error_rate.load_data(
        X,
        pd.Series([0, 1], index=X.index),
        sensitive_features=pd.Series([0, 1], index=X.index),
    )

    result = error_rate.gamma(lambda data: pd.Series([1, 0], index=data.index))

    assert result.error == 1.0


def test_error_rate_polars_compatible():
    """ErrorRate supports Polars input and Polars predictor output."""
    pl = pytest.importorskip("polars")
    X, y, sensitive_features = _classification_data()
    X_pl = pl.DataFrame(X)
    costs = {"fp": 0.1, "fn": 0.9}
    baseline = ErrorRate(costs=costs)
    baseline.load_data(X, y, sensitive_features=sensitive_features)
    expected = baseline.gamma(HistGradientBoostingClassifier().fit(X, y).predict)
    classifier = HistGradientBoostingClassifier().fit(X_pl, pl.Series(y))
    error_rate = ErrorRate(costs=costs)
    error_rate.load_data(
        X_pl,
        pl.Series(y),
        sensitive_features=pl.Series(sensitive_features),
    )

    error_pl = error_rate.gamma(lambda data: pl.Series(classifier.predict(data)))

    assert error_pl == expected


def test_error_rate_pyarrow_compatible():
    """ErrorRate supports PyArrow input and PyArrow predictor output."""
    pa = pytest.importorskip("pyarrow")
    X, y, sensitive_features = _classification_data()
    X_pa = pa.Table.from_pandas(pd.DataFrame(X))
    costs = {"fp": 0.1, "fn": 0.9}
    baseline = ErrorRate(costs=costs)
    baseline.load_data(X, y, sensitive_features=sensitive_features)
    expected = baseline.gamma(HistGradientBoostingClassifier().fit(X, y).predict)
    classifier = HistGradientBoostingClassifier().fit(X_pa, pa.array(y))
    error_rate = ErrorRate(costs=costs)
    error_rate.load_data(
        X_pa,
        pa.array(y),
        sensitive_features=pa.array(sensitive_features),
    )

    error_pa = error_rate.gamma(lambda data: pa.array(classifier.predict(data)))

    assert error_pa == expected
