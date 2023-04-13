# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import copy
import pickle
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from fairlearn.reductions import (
    DemographicParity,
    TruePositiveRateParity,
    FalsePositiveRateParity,
    ErrorRateParity,
)
from fairlearn.reductions import BoundedGroupLoss
from fairlearn.reductions import AbsoluteLoss, SquareLoss, ZeroOneLoss
from fairlearn.reductions import ExponentiatedGradient

from .test_utilities import _get_data


classification_moments = [
    DemographicParity,
    TruePositiveRateParity,
    FalsePositiveRateParity,
    ErrorRateParity,
]

bgl_losses = [AbsoluteLoss(0, 1), SquareLoss(0, 1), ZeroOneLoss()]


@pytest.mark.parametrize("moment", classification_moments)
def test_pickle_classification(moment):
    expgrad = ExponentiatedGradient(
        LogisticRegression(), constraints=moment(), eps=0.01, nu=0.01, max_iter=50
    )
    X, y, A = _get_data(A_two_dim=False, y_as_scores=False)

    expgrad.fit(X, y, sensitive_features=A)

    # The following should not throw an exception
    pickle.dumps(expgrad)


@pytest.mark.parametrize("loss", bgl_losses)
def test_pickle_bgl(loss):
    expgrad = ExponentiatedGradient(
        LinearRegression(),
        constraints=BoundedGroupLoss(loss=copy.deepcopy(loss), upper_bound=0.2),
        eps=0.01,
        nu=0.01,
        max_iter=50,
    )
    X, y, A = _get_data(A_two_dim=False, y_as_scores=True)

    expgrad.fit(X, y, sensitive_features=A)

    # The following should not throw an exception
    pickle.dumps(expgrad)
