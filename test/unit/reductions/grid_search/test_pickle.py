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
from fairlearn.reductions import GridSearch

from .utilities import _quick_data


classification_moments = [
    DemographicParity,
    TruePositiveRateParity,
    FalsePositiveRateParity,
    ErrorRateParity,
]

bgl_losses = [AbsoluteLoss(0, 1), SquareLoss(0, 1), ZeroOneLoss()]


@pytest.mark.parametrize("moment", classification_moments)
def test_pickle_classification(moment):
    gs = GridSearch(LogisticRegression(), constraints=moment())
    X, y, A = _quick_data()

    gs.fit(X, y, sensitive_features=A)

    # The following should not throw an exception
    pickle.dumps(gs)


@pytest.mark.parametrize("loss", bgl_losses)
def test_pickle_bgl(loss):
    gs = GridSearch(
        LinearRegression(),
        constraints=BoundedGroupLoss(loss=copy.deepcopy(loss), upper_bound=0.2),
    )
    X, y, A = _quick_data()

    gs.fit(X, y, sensitive_features=A)

    # The following should not throw an exception
    pickle.dumps(gs)
