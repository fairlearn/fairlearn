# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fairlearn.grid_search as gs

import numpy as np
import pandas as pd

import pytest

class LeastSquaresLearner:
    def __init__(self):
        self.weights = None

    def fit(self, X, Y, sample_weight):
        sqrtW = np.sqrt(sample_weight)
        matX = np.array(X) * sqrtW[:, np.newaxis]
        vecY = Y * sqrtW
        self.lsqinfo = np.linalg.lstsq(matX, vecY, rcond=-1)
        self.weights = pd.Series(self.lsqinfo[0], index=list(X))

    def predict(self, X):
        pred = X.dot(self.weights)
        return 1*(pred > 0.5)

class TestGridSearch:
    def _quick_data(self):
        feature_1 = [int(x) for x in '0110101' '0111101' '001011']
        feature_2 = [int(x) for x in '0000100' '0000011' '111111']
        feature_3 = [int(x) for x in '1111111' '1111111' '111111']
        X = pd.DataFrame({"feat1": feature_1, "feat2": feature_2, "feat3": feature_3})
        Y = pd.Series([int(x) for x in '0110100' '0010111' '001111'])
        A = pd.Series([int(x) for x in '0000000' '1111111' '010101'])
        return X, Y, A

    def test_generate_p0_p1_smoke(self):
        Y = [int(x) for x in '0110101' '0111101' '001011']

        target = gs.BinaryClassificationGridSearch()

        p0, p1 = target._generate_p0_p1(Y)

        assert p0 == 8/20
        assert p1 == 12/20

    def test_generate_weights(self):
        # Set up sample data
        A = [0, 0, 1, 1]
        Y = [0, 1, 0, 1]
        L = 10
        p_ratio = 2

        # Expected results
        W_expect = [ -21, -19, 9, 11]

        target = gs.BinaryClassificationGridSearch()
        W = target._generate_weights(Y, A, L, p_ratio)

        assert np.array_equal(W_expect, W), str(W)+str(W_expect)

    def test_grid_smoke(self):
        X, Y, A = self._quick_data()
        target = gs.BinaryClassificationGridSearch()
        result = target.grid_search_binary_protected_attribute(LeastSquaresLearner(), X, Y, A, number_lagrange_multipliers=11)
        assert len(result) == 11
        assert result[5]["lagrange_multiplier"] == 0
        multipliers = [x["lagrange_multiplier"] for x in result]
        assert len(np.unique(multipliers)) == len(result)
        assert sorted(multipliers)

    def test_grid_bad_lagrange_multiplier_specification(self):
        X, Y, A = self._quick_data()
        message = r"Must specify either lagrange_multipliers or number_lagrange_multipliers"

        target = gs.BinaryClassificationGridSearch()
        with pytest.raises(RuntimeError, match=message):
            _ = target.grid_search_binary_protected_attribute(LeastSquaresLearner(), X, Y, A, None, None)
        with pytest.raises(RuntimeError, match=message):
            _ = target.grid_search_binary_protected_attribute(LeastSquaresLearner(), X, Y, A, np.random.randint(10, size=3), 3)

    def test_grid_bad_A_labels(self):
        X, Y, _ = self._quick_data()
        message = r"Supplied protected_attribute labels not 0 or 1"

        bad_protected_attribute = pd.Series([int(x) for x in '0000000' '2222222' '020202'])

        target = gs.BinaryClassificationGridSearch()
        with pytest.raises(RuntimeError, match=message):
            _ = target.grid_search_binary_protected_attribute(LeastSquaresLearner(), X, Y, bad_protected_attribute)

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

        target = gs.BinaryClassificationGridSearch()
        result = target.grid_search_binary_protected_attribute(LeastSquaresLearner(), X, Y, A, number_lagrange_multipliers=5)
        assert len(result)==5

        # Check the weights for each of the models returned against the expected values
        # Note that the 'middle' value in each array corresponds to the Lagrange multiplier being zero
        expected_actual_feature_weights = [0, 0, 1.5, 0, 0]
        expected_protected_attribute_feature_weights = [-1, -1, 0, 1, 1]
        expected_constant_ones_feature_weights = [1, 1, -0.25, 0, 0]
        assert np.allclose(expected_actual_feature_weights, [x["model"].weights["actual_feature"] for x in result])
        assert np.allclose(expected_protected_attribute_feature_weights, [x["model"].weights["protected_attribute_feature"] for x in result])
        assert np.allclose(expected_constant_ones_feature_weights, [x["model"].weights["constant_ones_feature"] for x in result])