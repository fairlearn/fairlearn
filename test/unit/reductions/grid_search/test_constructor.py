# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from sklearn.linear_model import LinearRegression

from fairlearn.reductions import GridSearch
import fairlearn.reductions.moments as moments


@pytest.mark.parametrize("bad_constraint", [1, "random_string", LinearRegression()])
def test_bad_constraints(bad_constraint):
    message = ("Unsupported constraint. Please supply a Moment object "
               "or string from ['demographic_parity', 'bounded_group_loss']")
    with pytest.raises(ValueError) as execInfo:
        GridSearch(LinearRegression(), bad_constraint)
    assert message == execInfo.value.args[0]


def test_bad_selection_rule():
    message = "Unsupported selection rule"
    with pytest.raises(ValueError) as execInfo:
        GridSearch(LinearRegression(), moments.DemographicParity(), selection_rule="BAD")
    assert message == execInfo.value.args[0]


@pytest.mark.parametrize("weight", [0, 0.1, 0.5, 0.9, 1.0])
def test_allowed_constraint_weight(weight):
    GridSearch(LinearRegression(), moments.DemographicParity(), constraint_weight=weight)


@pytest.mark.parametrize("bad_weight", [-0.0001, 1.00001, -10, 10])
def test_bad_constraint_weight(bad_weight):
    message = "Must specify constraint_weight between 0.0 and 1.0"
    with pytest.raises(ValueError) as execInfo:
        GridSearch(LinearRegression(), moments.DemographicParity(), constraint_weight=bad_weight)
    assert message == execInfo.value.args[0]
