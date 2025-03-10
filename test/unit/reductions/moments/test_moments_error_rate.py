# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import narwhals as nw
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import pandas as pd
import polars as pl
import pytest

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


# TODO: at the moment only tests compatibility between pandas and polars; should also 
# test pyarrow
# TODO: parametrize test
def test_error_rate_narwhals_compatible():
    """Test that ErrorRate is compatible with any types passed via narwhals, that 
    returns are the same for all the methods and that the input types are returned."""

    # make up data for testing
    rng = np.random.default_rng(42)
    X, y = make_classification(n_features=10, class_sep=0.1, random_state=42)
    X[:, -1] = rng.integers(0, 2, size=(X.shape[0],))
    sensitive_features = X[:, -1]
    X_pd, y_pd, sensitive_features_pd = pd.DataFrame(X), pd.Series(y), pd.Series(sensitive_features)
    X_pl, y_pl, sensitive_features_pl = pl.DataFrame(X), pl.Series(y), pl.Series(sensitive_features)
    costs = {"fp":0.1, "fn":0.9}

    # we cannot check the method on the same instance, because data cannot be loaded
    # twice to the same instance)
    errorrate_np = ErrorRate(costs=costs)
    errorrate_pd = ErrorRate(costs=costs)
    errorrate_pl = ErrorRate(costs=costs)

    # check `ErrorRate.load_data()` works with all input types
    errorrate_np.load_data(X, y, sensitive_features=sensitive_features)
    errorrate_pd.load_data(X_pd, y_pd, sensitive_features=sensitive_features_pd)
    errorrate_pl.load_data(X_pl, y_pl, sensitive_features=sensitive_features_pl)

    # check `ErrorRate.gamma()` returns similar returns for all input types
    classifier = LogisticRegression().fit(X, y)
    error_np = errorrate_np.gamma(classifier.predict)
    assert isinstance(error_np, pd.Series)

    classifier = LogisticRegression().fit(X_pd, y_pd)
    error_pd = errorrate_pd.gamma(classifier.predict)
    assert isinstance(error_pd, pd.Series)

    classifier = LogisticRegression().fit(X_pl, y_pl)
    error_pl = errorrate_pl.gamma(classifier.predict)
    assert isinstance(error_pl, pl.Series)

    assert np.array_equal(error_np.to_numpy(), error_pd.to_numpy(), error_pl.to_numpy())
