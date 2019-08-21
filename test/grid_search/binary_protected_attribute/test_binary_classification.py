# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fairlearn.grid_search.binary_protected_attribute.binary_classification as bc

import numpy as np
import pandas as pd

from test import simple_learners

import pytest


class TestBinaryProtectedAttribute:
    def _quick_data(self, number_samples=8):
        feature_1 = np.random.randint(2, size=number_samples)
        feature_2 = np.random.randint(6, size=number_samples)
        feature_3 = np.random.randint(3, size=number_samples)
        X = np.stack((feature_1, feature_2, feature_3), -1).tolist()
        Y = np.random.randint(2, size=number_samples).tolist()
        A = np.random.randint(2, size=number_samples).tolist()
        return X, Y, A

    def _simple_threshold_data(self,
                               number_a0, number_a1,
                               a0_threshold, a1_threshold,
                               a0_label, a1_label):

        a0s = np.full(number_a0, a0_label)
        a1s = np.full(number_a1, a1_label)

        a0_scores = np.linspace(0, 1, number_a0)
        a1_scores = np.linspace(0, 1, number_a1)
        score_feature = np.concatenate((a0_scores, a1_scores), axis=None)

        A = np.concatenate((a0s, a1s), axis=None)

        Y_a0 = [x > a0_threshold for x in a0_scores]
        Y_a1 = [x > a1_threshold for x in a1_scores]

        Y = np.concatenate((Y_a0, Y_a1), axis=None)

        X = pd.DataFrame({"actual_feature": score_feature,
                          "protected_attribute_feature": A,
                          "constant_ones_feature": np.ones(len(Y))})
        return X, Y, A

    def _smoke_core(self, X, Y, A):
        result = bc.demographic_parity(
            simple_learners.LeastSquaresBinaryClassifierLearner(),
            X, Y, A,
            number_lagrange_multipliers=11)
        assert len(result) == 11

        # 'Central' multiplier of generated set should be 0
        # In practice, we might get a small error
        assert abs(result[5]["lagrange_multiplier"]) < 1e-15

        # Generated multipliers should be unique and in ascending order
        multipliers = [x["lagrange_multiplier"] for x in result]
        assert len(np.unique(multipliers)) == len(result)
        assert sorted(multipliers)

    def test_grid_smoke(self):
        X, Y, A = self._quick_data()
        self._smoke_core(X, Y, A)

    def test_grid_smoke_Y_numpy(self):
        X, Y, A = self._quick_data()
        Y_numpy = np.array(Y)
        self._smoke_core(X, Y_numpy, A)

    def test_grid_smoke_Y_pandas(self):
        X, Y, A = self._quick_data()
        Y_pandas = pd.Series(Y)
        self._smoke_core(X, Y_pandas, A)

    def test_grid_smoke_A_numpy(self):
        X, Y, A = self._quick_data()
        A_numpy = np.array(A)
        self._smoke_core(X, Y, A_numpy)

    def test_grid_smoke_A_pandas(self):
        X, Y, A = self._quick_data()
        A_pandas = pd.Series(A)
        self._smoke_core(X, Y, A_pandas)

    def test_grid_smoke_A_non_numeric(self):
        X, Y, _ = self._quick_data(number_samples=8)
        A = ["XY", "GF", "XY", "XY", "GF", "GF", "GF", "GF"]
        self._smoke_core(X, Y, A)

    def test_grid_smoke_X_numpy(self):
        X, Y, A = self._quick_data()
        X_numpy = np.array(X)
        self._smoke_core(X_numpy, Y, A)

    def test_grid_smoke_X_pandas(self):
        X, Y, A = self._quick_data()
        X_pandas = pd.DataFrame(
            X, columns=["feature_1", "feature_2", "feature_3"])
        self._smoke_core(X_pandas, Y, A)