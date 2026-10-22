# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from fairlearn.reductions import (
    BoundedGroupLoss,
    DemographicParity,
    EqualizedOdds,
    ExponentiatedGradient,
    SquareLoss,
)


@pytest.mark.parametrize(
    "moment",
    [DemographicParity(), EqualizedOdds(), BoundedGroupLoss(SquareLoss(0, 1))],
    ids=["demographic-parity", "equalized-odds", "bounded-group-loss"],
)
def test_moment_basis_growth_without_deprecation(moment):
    """Building constraint bases must not enlarge Series through scalar-only indexers."""
    X = np.arange(24).reshape(12, 2)
    y = np.tile([0, 1, 1, 0], 3)
    groups = np.tile([0, 0, 1, 1], 3)
    moment.load_data(X, y, sensitive_features=groups)
    assert len(moment.neg_basis_present) == moment.pos_basis.shape[1]


def test_exponentiated_gradient_series_growth_without_deprecation():
    """New predictors and mixture weights must work with warnings treated as errors."""
    X = np.arange(24).reshape(12, 2)
    y = np.tile([0, 1, 1, 0], 3)
    groups = np.tile([0, 0, 1, 1], 3)
    mitigator = ExponentiatedGradient(
        DecisionTreeClassifier(max_depth=1, random_state=0),
        DemographicParity(),
        max_iter=5,
    )
    mitigator.fit(X, y, sensitive_features=groups)
    assert len(mitigator.predictors_) == len(mitigator.weights_)
    assert mitigator.weights_.sum() == pytest.approx(1)
    assert mitigator.predict(X, random_state=0).shape == y.shape
