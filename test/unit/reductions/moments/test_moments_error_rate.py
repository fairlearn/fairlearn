# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import narwhals as nw
import numpy as np
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


# TODO: at the moment only tests compatibility between pandas and polars; should test 
# all narwhals compatible types?
# TODO: add control_features to tests
def test_error_rate_narwhals_compatible():
    """Test that ErrorRate is compatible with any types passed via narwhals, that 
    returns are the same for all the methods and that the input types are returned."""

    # prepare data for test    
    rng = np.random.default_rng(42)
    X, y = make_classification(n_features=10, class_sep=0.1, random_state=42)
    X[:, -1] = rng.integers(0, 2, size=(X.shape[0],))
    sensitive_features = X[:, -1]
    X_pd, y_pd, sensitive_features_pd = pd.DataFrame(X), pd.DataFrame(y), pd.DataFrame(sensitive_features)
    X_pl, y_pl, sensitive_features_pl = pl.DataFrame(X), pl.DataFrame(y), pl.DataFrame(sensitive_features)
    costs = {"fp":0.1, "fn":0.9}

    # check `ErrorRate.load_data()` works in all input types (remark: we cannot check
    # the method on the same instance, because data cannot be loaded twice to the same
    # instance)
    ErrorRate(costs=costs).load_data(X, y, sensitive_features=sensitive_features)
    ErrorRate(costs=costs).load_data(X_pd, y_pd, sensitive_features=sensitive_features_pd)
    ErrorRate(costs=costs).load_data(X_pl, y_pl, sensitive_features=sensitive_features_pl)

    # check `ErrorRate.gamma()`
    classifier = LogisticRegression().fit(X, y) # TODO: check if fairlearn has a dummy classifier for testing purpose
    errorrate = ErrorRate(costs=costs)
    errorrate.load_data(X, y, sensitive_features=sensitive_features)
    errorrate.gamma(classifier.predict)

    # TODO: check if there is something like pd.testing.assert_frame_equal() for narwhals
    """aligned_array = aligned.to_numpy() if isinstance(aligned, nw.DataFrame) else aligned
    exp_array = exp.to_numpy() if isinstance(exp, nw.DataFrame) else exp
    assert np.array_equal(aligned_array, exp_array)"""