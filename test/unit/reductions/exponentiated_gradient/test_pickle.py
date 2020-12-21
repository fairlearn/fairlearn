# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pickle
import pytest
from sklearn.linear_model import LogisticRegression

from fairlearn.reductions import DemographicParity
from fairlearn.reductions import ExponentiatedGradient

from .test_utilities import _get_data


moments = [DemographicParity]


@pytest.mark.parametrize("moment", moments)
def test_pickle(moment):
    expgrad = ExponentiatedGradient(
        LogisticRegression(),
        constraints=moment(),
        eps=0.01,
        nu=0.01,
        max_iter=50)
    X, y, A = _get_data(A_two_dim=False, y_as_scores=False)

    expgrad.fit(X, y, sensitive_features=A)

    pickle.dumps(expgrad)
