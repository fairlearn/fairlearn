# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier

from fairlearn.reductions import ErrorRate
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


def test_error_rate_narwhals_compatible():
    """Test that `ErrorRate` is compatible with several dataframe backends via narwhals,
    that return values are the same across different backends and that they come in
    types from the dataframe backend used."""

    # make up data for testing
    rng = np.random.default_rng(42)
    X, y = make_classification(n_features=10, class_sep=0.1, random_state=42)
    X[:, -1] = rng.integers(0, 2, size=(X.shape[0],))
    sensitive_features = X[:, -1]
    X_pd, y_pd, sensitive_features_pd = (
        pd.DataFrame(X),
        pd.Series(y),
        pd.Series(sensitive_features),
    )
    X_pl, y_pl, sensitive_features_pl = (
        pl.DataFrame(X),
        pl.Series(y),
        pl.Series(sensitive_features),
    )
    X_pa, y_pa, sensitive_features_pa = (
        pa.Table.from_pandas(X_pd),
        pa.Array.from_pandas(y_pd),
        pa.Array.from_pandas(sensitive_features_pd),
    )
    costs = {"fp": 0.1, "fn": 0.9}

    # create an instance per backend, since data can only be loaded once
    errorrate_np = ErrorRate(costs=costs)
    errorrate_pd = ErrorRate(costs=costs)
    errorrate_pl = ErrorRate(costs=costs)
    errorrate_pa = ErrorRate(costs=costs)

    # check `ErrorRate.load_data()` works with all the backends
    errorrate_np.load_data(X, y, sensitive_features=sensitive_features)
    errorrate_pd.load_data(X_pd, y_pd, sensitive_features=sensitive_features_pd)
    errorrate_pl.load_data(X_pl, y_pl, sensitive_features=sensitive_features_pl)
    errorrate_pa.load_data(X_pa, y_pa, sensitive_features=sensitive_features_pa)

    # check `ErrorRate.gamma()` returns similar returns for all input types
    classifier = HistGradientBoostingClassifier().fit(X, y)
    error_np = errorrate_np.gamma(classifier.predict)
    assert isinstance(error_np, pd.Series)

    classifier = HistGradientBoostingClassifier().fit(X_pd, y_pd)
    error_pd = errorrate_pd.gamma(classifier.predict)
    assert isinstance(error_pd, pd.Series)

    classifier = HistGradientBoostingClassifier().fit(X_pl, y_pl)
    error_pl = errorrate_pl.gamma(classifier.predict)
    assert isinstance(error_pl, pl.Series)

    classifier = HistGradientBoostingClassifier().fit(X_pa, y_pa)
    error_pa = errorrate_pa.gamma(classifier.predict)
    assert isinstance(error_pa, (pa.Array, pa.ChunkedArray))

    assert np.array_equal(error_np.to_numpy(), error_pd.to_numpy())
    assert np.array_equal(error_np.to_numpy(), error_pl.to_numpy())
    assert np.array_equal(error_np.to_numpy(), error_pa.to_numpy())
