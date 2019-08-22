# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fairlearn.grid_search.binary_protected_attribute.regression as reg  # noqa: E501

import numpy as np
import pandas as pd

from test import simple_learners

import pytest


class TestRegression:

    def test_grid_smoke_linear_regression(self):
        number_samples = 20
        feature_1 = np.random.randint(20, size=number_samples)
        feature_2 = np.random.randint(60, size=number_samples)
        feature_3 = np.random.randint(30, size=number_samples)
        X = np.stack((feature_1, feature_2, feature_3), -1).tolist()
        Y = np.random.randint(20, size=number_samples).tolist()
        A = np.random.randint(2, size=number_samples).tolist()

        result = reg.bounded_group_loss(
            simple_learners.LeastSquaresRegressor(),
            X, Y, A
        )
        assert len(result) == 11
