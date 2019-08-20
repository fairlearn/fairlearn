# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fairlearn.grid_search as gs

import numpy as np
import pandas as pd

import simple_learners

import pytest

class TestBinaryProtectedAttributeDemographicParity:
    def _quick_data(self, number_samples=8):
        feature_1 = np.random.randint(2, size=number_samples)
        feature_2 = np.random.randint(6, size=number_samples)
        feature_3 = np.random.randint(3, size=number_samples)
        X = np.stack((feature_1, feature_2, feature_3), -1).tolist() # pd.DataFrame({"feature_1": feature_1, "feature_2": feature_2, "feature_3": feature_3})
        Y = np.random.randint(2, size=number_samples).tolist()
        A = np.random.randint(2, size=number_samples).tolist()
        return X, Y, A

    def _smoke_core(self, X, Y, A):
        target = gs.BinaryProtectedAttributeDemographicParity()
        result = target.grid_search_binary_classification(simple_learners.LeastSquaresBinaryClassifierLearner(), X, Y, A, number_lagrange_multipliers=11)
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

    def test_grid_smoke_X_numpy(self):
        X, Y, A = self._quick_data()
        X_numpy = np.array(X)
        self._smoke_core(X_numpy, Y, A)

    def test_grid_smoke_X_pandas(self):
        X, Y, A = self._quick_data()
        X_pandas = pd.DataFrame(X, columns=["feature_1", "feature_2", "feature_3"])
        self._smoke_core(X_pandas, Y, A)

    def test_generate_p0_p1_smoke(self):
        number_labels = 32
        Y = np.random.randint(2, size=number_labels)

        target = gs.BinaryProtectedAttributeDemographicParity()

        p0, p1 = target._generate_p0_p1(Y)

        assert p0 == 1 - sum(Y) / number_labels
        assert p1 == sum(Y) / number_labels

    def test_generate_weights(self):
        # Set up sample data
        A = [0, 0, 1, 1]
        Y = [0, 1, 0, 1]
        L = 10
        p_ratio = 2

        # Expected results
        W_expect = [ -21, -19, 9, 11]

        target = gs.BinaryProtectedAttributeDemographicParity()
        W = target._generate_weights(Y, A, L, p_ratio)

        assert np.array_equal(W_expect, W), str(W)+str(W_expect)

    def test_grid_bad_lagrange_multiplier_specification(self):
        X, Y, A = self._quick_data()
        message = r"Must specify either lagrange_multipliers or number_lagrange_multipliers"

        target = gs.BinaryProtectedAttributeDemographicParity()
        with pytest.raises(RuntimeError, match=message):
            _ = target.grid_search_binary_classification(simple_learners.LeastSquaresBinaryClassifierLearner(), X, Y, A, None, None)
        with pytest.raises(RuntimeError, match=message):
            _ = target.grid_search_binary_classification(simple_learners.LeastSquaresBinaryClassifierLearner(), X, Y, A, np.random.randint(10, size=3), 3)

    def test_non_binary_protected_attribute(self):
        X, Y, _ = self._quick_data()
        message = r"Protected Attribute is not binary"

        bad_protected_attribute = [0, 1, 2]

        target = gs.BinaryProtectedAttributeDemographicParity()
        with pytest.raises(RuntimeError, match=message):
            _ = target.grid_search_binary_classification(simple_learners.LeastSquaresBinaryClassifierLearner(), X, Y, bad_protected_attribute)

    def test_non_binary_labels(self):
        X, _, A = self._quick_data(8)
        bad_labels = [0, 1, 2, 0, 1, 2, 0, 1]
        message = r"Supplied Y labels are not binary"

        target = gs.BinaryProtectedAttributeDemographicParity()
        with pytest.raises(RuntimeError, match=message):
            _ = target.grid_search_binary_classification(simple_learners.LeastSquaresBinaryClassifierLearner(), X, bad_labels, A)

    def test_labels_not_0_1(self):
        X, _, A = self._quick_data(8)
        bad_labels = [0, 2, 2, 0, 0, 2, 0, 0]
        message = r"Supplied Y labels are not 0 or 1"
        
        target = gs.BinaryProtectedAttributeDemographicParity()
        with pytest.raises(RuntimeError, match=message):
            _ = target.grid_search_binary_classification(simple_learners.LeastSquaresBinaryClassifierLearner(), X, bad_labels, A)

    def test_grid_already_fair(self):
        # Number of samples of each attribute to generate
        # Large numbers required to beat down the errors in the weights
        num_samples_each = 400000

        # Create a 'rigged' dataset which while it contains the protected attribute, actually doesn't depend on it

        # Attribute array is evenly divided between zeros and ones
        A = pd.Series(np.concatenate((np.zeros(num_samples_each), np.ones(num_samples_each)), axis=None))

        # Generate a set of scores which we threshold to get the label
        scores = np.linspace(0, 1, num_samples_each)

        # Need two copies of the scores, one for each value of the protected features
        actual_feature = pd.Series(np.concatenate((scores, scores), axis=None))

        # Generate the labels by thresholding
        Y = pd.Series([x>0.5 for x in actual_feature])

        # Assemble the dataframe of samples
        # Need a extra 'ones' column to allow our simple least squares learner to work properly
        X = pd.DataFrame({"actual_feature": actual_feature, "protected_attribute_feature": A, "constant_ones_feature": np.ones(len(Y))})

        target = gs.BinaryProtectedAttributeDemographicParity()
        result = target.grid_search_binary_classification(simple_learners.LeastSquaresBinaryClassifierLearner(), X, Y, A, number_lagrange_multipliers=5)
        assert len(result)==5

        # Check the weights for each of the models returned against the expected values
        # Note that the 'middle' value in each array corresponds to the Lagrange multiplier being zero
        expected_actual_feature_weights = [0, 0, 1.5, 0, 0]
        expected_protected_attribute_feature_weights = [-1, -1, 0, 1, 1]
        expected_constant_ones_feature_weights = [1, 1, -0.25, 0, 0]
        assert np.allclose(expected_actual_feature_weights, [x["model"].weights["actual_feature"] for x in result])
        assert np.allclose(expected_protected_attribute_feature_weights, [x["model"].weights["protected_attribute_feature"] for x in result])
        assert np.allclose(expected_constant_ones_feature_weights, [x["model"].weights["constant_ones_feature"] for x in result])