# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity, ErrorRate
from fairlearn.reductions._exponentiated_gradient._constants import _MIN_ITER

import lightgbm as lgb


_PRECISION = 1e-6

sensitive_features = [str(x) for x in 'AAAAAAA' 'BBBBBBB' 'CCCCCC']
labels =             [int(x) for x in '0110100' '0010111' '001111']  # noqa: E222
X1 =                 [int(x) for x in '0110101' '0111101' '001011']  # noqa: E222
X2 =                 [int(x) for x in '0000100' '0000011' '111111']  # noqa: E222
X3 =                 [int(x) for x in '1111111' '1111111' '111111']  # noqa: E222
scores =          [int(x)/10 for x in '0001124' '0223346' '134447']  # noqa: E222


def _get_data(A_two_dim=False, y_as_scores=False, flip_y=False):
    X = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})

    if y_as_scores:
        y = pd.Series(scores)
    else:  # classification
        y = pd.Series(labels)
        if flip_y:
            y = pd.Series([int(not i) for i in labels])

    if A_two_dim:
        # Stacking the same column a few times will result in the identical groups
        # compared to using a single column, therefore results should be the same.
        A = np.stack((sensitive_features, sensitive_features), -1)
    else:
        A = pd.Series(sensitive_features)
    return X, y, A

smoke_test_data = [
        {"constraint_class": DemographicParity, "eps": 0.100,
         "best_gap": 0.000000, "last_iter": 5,
         "best_iter": 5, "disp": 0.100000,
         "error": 0.250000, "n_oracle_calls": 34, "n_oracle_calls_dummy_returned": 0,
         "n_predictors": 1},
        {"constraint_class": DemographicParity, "eps": 0.100,
         "best_gap": 0.000000, "last_iter": 5,
         "best_iter": 5, "disp": -0.020000,
         "error": 0.250000, "n_oracle_calls": 16, "n_oracle_calls_dummy_returned": 12,
         "n_predictors": 1, "ratio": 0.8},
]

def run_smoke_test_binary_classification(data, flipped=False):
    lgb_params = {
        'objective' : 'binary',
        'metric' : 'auc',
        'learning_rate': 0.03,
        'num_leaves' : 10,
        'max_depth' : 3
    }
    learner = lgb.LGBMClassifier(**lgb_params)

    if "ratio" in data.keys():
        disparity_moment = data["constraint_class"](
            ratio_bound_slack=data["eps"],
            ratio_bound=data["ratio"])
    else:
        disparity_moment = data["constraint_class"](
            difference_bound=data["eps"])

    # Create Exponentiated Gradient object with a copy of the constraint.
    # The original disparity_moment object is used for validation, so the
    # assumption is that the moment logic is correct in these tests.
    expgrad = ExponentiatedGradient(
        learner,
        constraints=deepcopy(disparity_moment),
        eps=data["eps"])

    X, y, A = _get_data(A_two_dim=False, flip_y=flipped)

    expgrad.fit(X, y, sensitive_features=A)

    _assert_expgrad_state(expgrad, data)

    # select probability of predicting 1
    def Q(X): return expgrad._pmf_predict(X)[:, 1]
    default_objective = ErrorRate()
    disparity_moment.load_data(X, y, sensitive_features=A)
    default_objective.load_data(X, y, sensitive_features=A)
    disparity = disparity_moment.gamma(Q).max()
    error = default_objective.gamma(Q)[0]
    assert disparity == pytest.approx(data["disp"], abs=_PRECISION)
    assert error == pytest.approx(data["error"], abs=_PRECISION)

@pytest.mark.parametrize("testdata", smoke_test_data)
def test_smoke(testdata):
    run_smoke_test_binary_classification(testdata)

def _assert_expgrad_state(expgrad, data):
    n_predictors = len(expgrad.predictors_)
    assert expgrad.best_gap_ == pytest.approx(data["best_gap"], abs=_PRECISION)
    assert expgrad.best_gap_ < expgrad.nu
    assert expgrad.last_iter_ == data["last_iter"]
    assert expgrad.best_iter_ == data["best_iter"]
    assert expgrad.last_iter_ >= _MIN_ITER
    assert expgrad.n_oracle_calls_ == data["n_oracle_calls"]
    assert expgrad.n_oracle_calls_dummy_returned_ == data["n_oracle_calls_dummy_returned"]
    assert n_predictors == data["n_predictors"]
    assert len(expgrad.oracle_execution_times_) == expgrad.n_oracle_calls_