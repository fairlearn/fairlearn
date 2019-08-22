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

    def _simple_regression_data(self,
                                number_a0, number_a1,
                                a0_factor, a1_factor,
                                a0_label, a1_label):

        a0s = np.full(number_a0, a0_label)
        a1s = np.full(number_a1, a1_label)

        a0_scores = np.linspace(0, 1, number_a0)
        a1_scores = np.linspace(0, 1, number_a1)
        score_feature = np.concatenate((a0_scores, a1_scores), axis=None)

        A = np.concatenate((a0s, a1s), axis=None)

        Y_a0 = a0_factor * a0_scores
        Y_a1 = a1_factor * a1_scores

        Y = np.concatenate((Y_a0, Y_a1), axis=None)

        X = pd.DataFrame({"actual_feature": score_feature,
                          "protected_attribute_feature": A,
                          "constant_ones_feature": np.ones(len(Y))})
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

    def test_bgl_already_fair(self):
        a0_count = 10
        a1_count = 20

        a0_label = 3
        a1_label = 7

        a0_factor = 1
        a1_factor = 1

        X, Y, A = self._simple_regression_data(a0_count, a1_count,
                                               a0_factor, a1_factor,
                                               a0_label, a1_label)

        lsr = simple_learners.LeastSquaresRegressor()

        result = reg.bounded_group_loss(lsr,
                                        X, Y, A,
                                        number_of_tradeoffs=5)
        assert len(result) == 5

        result_actual_feature_weights = [
            x["model"].weights["actual_feature"] for x in result]
        result_protected_attribute_feature_weights = [
            x["model"].weights["protected_attribute_feature"] for x in result]
        result_constant_ones_feature_weights = [
            x["model"].weights["constant_ones_feature"] for x in result]

        expected_actual_feature_weights = [1, 1, 1, 1, 1]
        expected_protected_attribute_feature_weights = [0, 0, 0, 0, 0]
        expected_constant_ones_feature_weights = [0, 0, 0, 0, 0]

        assert np.allclose(result_actual_feature_weights,
                           expected_actual_feature_weights)
        assert np.allclose(result_protected_attribute_feature_weights,
                           expected_protected_attribute_feature_weights)
        assert np.allclose(result_constant_ones_feature_weights,
                           expected_constant_ones_feature_weights)
