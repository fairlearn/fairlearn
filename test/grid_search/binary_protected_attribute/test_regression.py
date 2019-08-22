# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fairlearn.grid_search.binary_protected_attribute.regression as reg  # noqa: E501

import numpy as np
import pandas as pd

from test import simple_learners

import pytest


class TestRegression:
    def _quick_data(self, number_samples=8):
        feature_1 = np.random.rand(number_samples)
        feature_2 = np.random.rand(number_samples)
        feature_3 = np.random.rand(number_samples)
        X = np.stack((feature_1, feature_2, feature_3), -1).tolist()

        fuzz = np.random.rand(number_samples)
        Y = (feature_1 + 2 * feature_2 + 3 * feature_3 + fuzz).tolist()
        A = np.random.randint(2, size=number_samples).tolist()
        return X, Y, A

    def _smoke_bgl_core(self, X, Y, A):
        result = reg.bounded_group_loss(
            simple_learners.LeastSquaresRegressor(),
            X, Y, A
        )
        assert len(result) == 11

    def test_bgl_smoke(self):
        X, Y, A = self._quick_data()
        self._smoke_bgl_core(X, Y, A)

    def test_bgl_smoke_Y_numpy(self):
        X, Y, A = self._quick_data()
        Y_numpy = np.array(Y)
        self._smoke_bgl_core(X, Y_numpy, A)

    def test_bgl_smoke_Y_pandas(self):
        X, Y, A = self._quick_data()
        Y_pandas = pd.Series(Y)
        self._smoke_bgl_core(X, Y_pandas, A)

    def test_bgl_smoke_A_numpy(self):
        X, Y, A = self._quick_data()
        A_numpy = np.array(A)
        self._smoke_bgl_core(X, Y, A_numpy)

    def test_bgl_smoke_A_pandas(self):
        X, Y, A = self._quick_data()
        A_pandas = pd.Series(A)
        self._smoke_bgl_core(X, Y, A_pandas)

    def test_bgl_smoke_A_nonnumeric(self):
        X, Y, _ = self._quick_data(4)
        A = ["AB", "AB", "XF", "AB"]
        self._smoke_bgl_core(X, Y, A)

    def test_bgl_smoke_X_numpy(self):
        X, Y, A = self._quick_data()
        X_numpy = np.array(X)
        self._smoke_bgl_core(X_numpy, Y, A)

    def test_bgl_smoke_X_pandas(self):
        X, Y, A = self._quick_data()
        X_pandas = pd.DataFrame(
            X, columns=["feature_1", "feature_2", "feature_3"])
        self._smoke_bgl_core(X_pandas, Y, A)

    def test_bgl_bad_tradeoff_specification(self):
        X, Y, A = self._quick_data()
        message = str("Must specify either tradeoffs "
                      "or number_of_tradeoffs")

        with pytest.raises(RuntimeError, match=message):
            _ = reg.bounded_group_loss(
                simple_learners.LeastSquaresRegressor(),
                X, Y, A,
                None, None)
        with pytest.raises(RuntimeError, match=message):
            _ = reg.bounded_group_loss(
                simple_learners.LeastSquaresRegressor(),
                X, Y, A,
                np.random.randint(10, size=3), 3)
