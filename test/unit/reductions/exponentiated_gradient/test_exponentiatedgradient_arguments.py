# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest


from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity
from fairlearn.reductions import ErrorRate
from .simple_learners import LeastSquaresBinaryClassifierLearner
from .test_utilities import sensitive_features, X1, X2, X3, labels

from test.unit.input_convertors import conversions_for_1d, ensure_ndarray, \
    ensure_dataframe, _map_into_single_column
from test.unit.reductions.conftest import is_invalid_transformation

# ===============================================================

# Ways of transforming the data
candidate_X_transforms = [ensure_ndarray, ensure_dataframe]
candidate_Y_transforms = conversions_for_1d
candidate_A_transforms = conversions_for_1d

# ================================================================

_PRECISION = 1e-6


def _get_data(A_two_dim=False):
    X = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
    y = pd.Series(labels)

    if A_two_dim:
        # Stacking the same column a few times will result in the identical groups
        # compared to using a single column, therefore results should be the same.
        A = np.stack((sensitive_features, sensitive_features), -1)
    else:
        A = pd.Series(sensitive_features)
    return X, y, A


class TestExponentiatedGradientArguments:
    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_argument_types(self, transformX, transformY, transformA, A_two_dim):
        # This is an expanded-out version of one of the smoke tests
        X, y, A = _get_data(A_two_dim)
        merged_A = _map_into_single_column(A)

        expgrad = ExponentiatedGradient(
            LeastSquaresBinaryClassifierLearner(),
            constraints=DemographicParity(),
            eps=0.1)
        expgrad.fit(transformX(X), transformY(y), sensitive_features=transformA(A))

        Q = expgrad._best_classifier
        n_classifiers = len(expgrad._classifiers)

        disparity_moment = DemographicParity()
        disparity_moment.load_data(X, y, sensitive_features=merged_A)
        error = ErrorRate()
        error.load_data(X, y, sensitive_features=merged_A)
        disparity = disparity_moment.gamma(Q).max()
        error = error.gamma(Q)[0]

        assert expgrad._best_gap == pytest.approx(0.0000, abs=_PRECISION)
        assert expgrad._last_t == 5
        assert expgrad._best_t == 5
        assert disparity == pytest.approx(0.1, abs=_PRECISION)
        assert error == pytest.approx(0.25, abs=_PRECISION)
        assert expgrad._n_oracle_calls == 32
        assert n_classifiers == 3
