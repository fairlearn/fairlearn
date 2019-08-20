# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fairlearn.grid_search as gs

import numpy as np
import pandas as pd

import simple_learners

import pytest


class TestBinaryProtectedAttribute:
    def _quick_data(self, number_samples=8):
        feature_1 = np.random.randint(2, size=number_samples)
        feature_2 = np.random.randint(6, size=number_samples)
        feature_3 = np.random.randint(3, size=number_samples)
        # pd.DataFrame({"feature_1": feature_1, "feature_2": feature_2, "feature_3": feature_3})
        X = np.stack((feature_1, feature_2, feature_3), -1).tolist()
        Y = np.random.randint(2, size=number_samples).tolist()
        A = np.random.randint(2, size=number_samples).tolist()
        return X, Y, A

    def _smoke_core(self, X, Y, A):
        target = gs.BinaryProtectedAttribute()
        result = target.demographic_parity_binary_classification(
            simple_learners.LeastSquaresBinaryClassifierLearner(), X, Y, A, number_lagrange_multipliers=11)
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

    def test_generate_protected_attribute_info_smoke(self):
        number_attributes = 32
        A = np.random.randint(2, size=number_attributes)

        target = gs.BinaryProtectedAttribute()

        p0, p1, a0_val = target._generate_protected_attribute_info(A)

        assert p0 == 1 - sum(A) / number_attributes
        assert p1 == sum(A) / number_attributes
        # Need to handle case were array is all 1s
        if p0 == 1:
            assert a0_val == A[0]
        else:
            assert a0_val == 0

    def test_generate_protected_attribute_info_non_numeric(self):
        A = ["AB", "AB", "CD", "AB"]

        target = gs.BinaryProtectedAttribute()

        p0, p1, a0_val = target._generate_protected_attribute_info(A)
        assert p0 == 0.75
        assert p1 == 0.25
        assert a0_val == "AB"

    def test_generate_weights_smoke(self):
        # Set up sample data
        A = [0, 0, 1, 1]
        Y = [0, 1, 0, 1]
        L = 10
        p_ratio = 2
        a0_val = 0  # Must match A

        # Expected results
        W_expect = [-21, -19, 9, 11]

        target = gs.BinaryProtectedAttribute()
        W = target._generate_weights(Y, A, L, p_ratio, a0_val)

        assert np.array_equal(W_expect, W), str(W) + str(W_expect)

    def test_grid_bad_lagrange_multiplier_specification(self):
        X, Y, A = self._quick_data()
        message = r"Must specify either lagrange_multipliers or number_lagrange_multipliers"

        target = gs.BinaryProtectedAttribute()
        with pytest.raises(RuntimeError, match=message):
            _ = target.demographic_parity_binary_classification(
                simple_learners.LeastSquaresBinaryClassifierLearner(), X, Y, A, None, None)
        with pytest.raises(RuntimeError, match=message):
            _ = target.demographic_parity_binary_classification(
                simple_learners.LeastSquaresBinaryClassifierLearner(), X, Y, A, np.random.randint(10, size=3), 3)

    def test_non_binary_protected_attribute(self):
        X, Y, _ = self._quick_data()
        message = r"Protected Attribute is not binary"

        bad_protected_attribute = [0, 1, 2]

        target = gs.BinaryProtectedAttribute()
        with pytest.raises(RuntimeError, match=message):
            _ = target.demographic_parity_binary_classification(
                simple_learners.LeastSquaresBinaryClassifierLearner(), X, Y, bad_protected_attribute)

    def test_non_binary_labels(self):
        X, _, A = self._quick_data(8)
        bad_labels = [0, 1, 2, 0, 1, 2, 0, 1]
        message = r"Supplied Y labels are not binary"

        target = gs.BinaryProtectedAttribute()
        with pytest.raises(RuntimeError, match=message):
            _ = target.demographic_parity_binary_classification(
                simple_learners.LeastSquaresBinaryClassifierLearner(), X, bad_labels, A)

    def test_labels_not_0_1(self):
        X, _, A = self._quick_data(8)
        bad_labels = [0, 2, 2, 0, 0, 2, 0, 0]
        message = r"Supplied Y labels are not 0 or 1"

        target = gs.BinaryProtectedAttribute()
        with pytest.raises(RuntimeError, match=message):
            _ = target.demographic_parity_binary_classification(
                simple_learners.LeastSquaresBinaryClassifierLearner(), X, bad_labels, A)

    def test_lagrange_multiplier_generation(self):
        # Check that the Lagrange multiplier values are being generated for the correct
        # range. We do not do full checks on np.linspace
        X, Y, A = self._quick_data(32)

        # Reimplement the multiplier computation
        p0 = 1 - (sum(A) / len(A))
        p1 = sum(A) / len(A)
        limit = 1
        if p0 / p1 > 1:
            limit = p0 / p1

        target = gs.BinaryProtectedAttribute()
        results = target.demographic_parity_binary_classification(
            simple_learners.LeastSquaresBinaryClassifierLearner(), X, Y, A, number_lagrange_multipliers=3)

        expected_multipliers = [-2 * limit, 0, 2 * limit]
        actual_multipliers = [r["lagrange_multiplier"] for r in results]

        assert np.allclose(expected_multipliers, actual_multipliers)

    @pytest.mark.parametrize("testdata", [[0, 1], [-1, 1], [-2, 4], [1, 2]])
    def test_grid_already_fair(self, testdata):
        # Number of samples of each attribute to generate
        # Large numbers required to beat down the errors in the weights
        num_samples_each = 400000

        # We need a numeric binary attribute, but it need not
        # be taken from {0, 1}
        a_offset = testdata[0]
        a_scale = testdata[1]
        assert a_scale != 0
        assert a_offset == int(a_offset)
        assert a_scale == int(a_scale)

        # Create a 'rigged' dataset which while it contains the protected attribute, actually doesn't depend on it

        # Attribute array is evenly divided between two values
        A_raw = pd.Series(np.concatenate(
            (np.zeros(num_samples_each), np.ones(num_samples_each)), axis=None))
        A = a_scale * A_raw + a_offset

        # Generate a set of scores which we threshold to get the label
        scores = np.linspace(0, 1, num_samples_each)

        # Need two copies of the scores, one for each value of the protected features
        actual_feature = pd.Series(np.concatenate((scores, scores), axis=None))

        # Generate the labels by thresholding
        Y = pd.Series([x > 0.5 for x in actual_feature])

        # Assemble the dataframe of samples
        # Need a extra 'ones' column to allow our simple least squares learner to work properly
        X = pd.DataFrame({"actual_feature": actual_feature,
                          "protected_attribute_feature": A, "constant_ones_feature": np.ones(len(Y))})

        target = gs.BinaryProtectedAttribute()
        result = target.demographic_parity_binary_classification(
            simple_learners.LeastSquaresBinaryClassifierLearner(), X, Y, A, number_lagrange_multipliers=5)
        assert len(result) == 5

        # Check the weights for each of the models returned against the expected values
        # Note that the 'middle' value in each array corresponds to the Lagrange multiplier being zero
        expected_actual_feature_weights = [0, 0, 1.5, 0, 0]
        expected_protected_attribute_feature_weights = np.array([-1, -1, 0, 1, 1]) / a_scale
        expected_constant_ones_feature_weights = [1, 1, -0.25, 0, 0]
        result_actual_feature_weights = [
            x["model"].weights["actual_feature"] for x in result]
        result_protected_attribute_feature_weights = [
            x["model"].weights["protected_attribute_feature"] for x in result]
        result_constant_ones_feature_weights = [
            x["model"].weights["constant_ones_feature"] for x in result]

        assert np.allclose(expected_actual_feature_weights,
                           result_actual_feature_weights, rtol=1e-3)
        assert np.allclose(expected_protected_attribute_feature_weights,
                           result_protected_attribute_feature_weights, rtol=1e-3)

        # Have not quite figured out functional form for constant_ones feature
        assert expected_constant_ones_feature_weights[2] == pytest.approx(result_constant_ones_feature_weights[2], rel=1e-3)
        if a_scale==1 and a_offset==0:
            # Do know what the unrescaled constant_ones feature weight should be
            assert np.allclose(expected_constant_ones_feature_weights, result_constant_ones_feature_weights)
