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
        target = gs.BinaryProtectedAttribute()
        result = target.demographic_parity_binary_classification(
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
        message = str("Must specify either lagrange_multipliers "
                      "or number_lagrange_multipliers")

        target = gs.BinaryProtectedAttribute()
        with pytest.raises(RuntimeError, match=message):
            _ = target.demographic_parity_binary_classification(
                simple_learners.LeastSquaresBinaryClassifierLearner(),
                X, Y, A,
                None, None)
        with pytest.raises(RuntimeError, match=message):
            _ = target.demographic_parity_binary_classification(
                simple_learners.LeastSquaresBinaryClassifierLearner(),
                X, Y, A,
                np.random.randint(10, size=3), 3)

    def test_non_binary_protected_attribute(self):
        X, Y, _ = self._quick_data()
        message = r"Protected Attribute is not binary"

        bad_protected_attribute = [0, 1, 2]

        target = gs.BinaryProtectedAttribute()
        with pytest.raises(RuntimeError, match=message):
            _ = target.demographic_parity_binary_classification(
                simple_learners.LeastSquaresBinaryClassifierLearner(),
                X, Y, bad_protected_attribute)

    def test_non_binary_labels(self):
        X, _, A = self._quick_data(8)
        bad_labels = [0, 1, 2, 0, 1, 2, 0, 1]
        message = r"Supplied Y labels are not binary"

        target = gs.BinaryProtectedAttribute()
        with pytest.raises(RuntimeError, match=message):
            _ = target.demographic_parity_binary_classification(
                simple_learners.LeastSquaresBinaryClassifierLearner(),
                X, bad_labels, A)

    def test_labels_not_0_1(self):
        X, _, A = self._quick_data(8)
        bad_labels = [0, 2, 2, 0, 0, 2, 0, 0]
        message = r"Supplied Y labels are not 0 or 1"

        target = gs.BinaryProtectedAttribute()
        with pytest.raises(RuntimeError, match=message):
            _ = target.demographic_parity_binary_classification(
                simple_learners.LeastSquaresBinaryClassifierLearner(),
                X, bad_labels, A)

    def test_lagrange_multiplier_generation(self):
        # Check that the Lagrange multiplier values are being
        # generated for the correct range. We do not do full checks
        # on np.linspace
        X, Y, A = self._quick_data(32)

        # Reimplement the multiplier computation
        p0 = 1 - (sum(A) / len(A))
        p1 = sum(A) / len(A)
        limit = 1
        if p0 / p1 > 1:
            limit = p0 / p1

        target = gs.BinaryProtectedAttribute()
        results = target.demographic_parity_binary_classification(
            simple_learners.LeastSquaresBinaryClassifierLearner(),
            X, Y, A,
            number_lagrange_multipliers=3)

        expected_multipliers = [-2 * limit, 0, 2 * limit]
        actual_multipliers = [r["lagrange_multiplier"] for r in results]

        assert np.allclose(expected_multipliers, actual_multipliers)

    @pytest.mark.parametrize("testdata", [[0, 1], [-1, 1], [-2, 4], [1, 2]])
    def test_grid_already_fair(self, testdata):
        # Number of samples of each attribute to generate
        # Large numbers required to beat down the errors in the weights
        num_samples_each = 400000

        # Give half the dataset a label 0 and the other half 1
        score_threshold = 0.5

        # We need a numeric binary attribute, but it need not
        # be taken from {0, 1}
        a_offset = testdata[0]
        a_scale = testdata[1]
        assert a_scale != 0
        assert a_offset == int(a_offset)
        assert a_scale == int(a_scale)

        a0_label = 0 * a_scale + a_offset
        a1_label = 1 * a_scale + a_offset

        X, Y, A = self._simple_threshold_data(num_samples_each,
                                              num_samples_each,
                                              score_threshold,
                                              score_threshold,
                                              a0_label,
                                              a1_label)

        target = gs.BinaryProtectedAttribute()
        result = target.demographic_parity_binary_classification(
            simple_learners.LeastSquaresBinaryClassifierLearner(),
            X, Y, A, number_lagrange_multipliers=5)
        assert len(result) == 5

        # Check the weights for each of the models returned against
        # the expected values
        # Note that the 'middle' value in each array corresponds to the
        # Lagrange multiplier being zero
        expected_actual_feature_weights = [0, 0, 1.5, 0, 0]
        expected_protected_attribute_feature_weights = np.array(
            [-1, -1, 0, 1, 1]) / a_scale
        expected_constant_ones_feature_weights = [1, 1, -0.25, 0, 0]
        result_actual_feature_weights = [
            x["model"].weights["actual_feature"] for x in result]
        result_protected_attribute_feature_weights = [
            x["model"].weights["protected_attribute_feature"] for x in result]
        result_constant_ones_feature_weights = [
            x["model"].weights["constant_ones_feature"] for x in result]

        assert np.allclose(expected_actual_feature_weights,
                           result_actual_feature_weights,
                           rtol=1e-3)
        assert np.allclose(expected_protected_attribute_feature_weights,
                           result_protected_attribute_feature_weights,
                           rtol=1e-3)

        # Have not quite figured out functional form for constant_ones feature
        assert expected_constant_ones_feature_weights[2] == pytest.approx(
            result_constant_ones_feature_weights[2], rel=1e-3)
        if a_scale == 1 and a_offset == 0:
            # For the unscaled case, we do know what the unrescaled
            # constant_ones feature weights should be
            assert np.allclose(
                expected_constant_ones_feature_weights,
                result_constant_ones_feature_weights)

    def test_grid_fair_uneven_populations(self):
        # Variant of test_grid_already_fair, which has unequal populations
        # in the two classes
        # Also allow the threshold to be adjustable

        score_threshold = 0.625

        number_a0 = 5000
        number_a1 = 200

        a0_label = 17
        a1_label = 37

        X, Y, A = self._simple_threshold_data(number_a0, number_a1,
                                              score_threshold, score_threshold,
                                              a0_label, a1_label)

        target = gs.BinaryProtectedAttribute()
        result = target.demographic_parity_binary_classification(
            simple_learners.LeastSquaresBinaryClassifierLearner(),
            X, Y, A, number_lagrange_multipliers=11)
        assert len(result) == 11

        result_actual_feature_weights = [
            x["model"].weights["actual_feature"] for x in result]
        result_protected_attribute_feature_weights = [
            x["model"].weights["protected_attribute_feature"] for x in result]
        result_constant_ones_feature_weights = [
            x["model"].weights["constant_ones_feature"] for x in result]

        # The following 'golden values' are empirically determined

        # See the actual feature become maximally important for
        # a multiplier of zero (as expected for a fair dataset)
        expected_actual_feature_weights = [
            0, 0, 0, 0.4993, 1.0264, 1.4057, 0.8190, 0.2876, 0, 0, 0]

        # The zero in the middle of this corresponds to a multiplier
        # of zero, and is expected since the dataset was created to
        # be fair
        expected_protected_attribute_feature_weights = [
            -0.05,
            -0.05,
            -0.05,
            -0.03604,
            -0.02264,
            0,
            0.0353,
            0.0445,
            0.0500,
            0.0500,
            0.0500]

        expected_constant_ones_feature_weights = [
            1.8500,
            1.8500,
            1.8500,
            1.0898,
            0.3481,
            -0.3279,
            -0.7351,
            -0.7921,
            -0.8500,
            -0.8500,
            -0.8500]

        assert np.allclose(result_actual_feature_weights,
                           expected_actual_feature_weights,
                           rtol=1e-3)
        assert np.allclose(result_protected_attribute_feature_weights,
                           expected_protected_attribute_feature_weights,
                           rtol=1e-3)
        assert np.allclose(result_constant_ones_feature_weights,
                           expected_constant_ones_feature_weights,
                           rtol=1e-3)

    def test_grid_unfair(self):
        # Variant of test_grid_already_fair where the two populations
        # have the same distribution but different thresholds

        number_a0 = 5000
        number_a1 = 200

        a0_threshold = 0.2
        a1_threshold = 0.8

        a0_label = 3
        a1_label = 7

        X, Y, A = self._simple_threshold_data(number_a0, number_a1,
                                              a0_threshold, a1_threshold,
                                              a0_label, a1_label)

        target = gs.BinaryProtectedAttribute()
        result = target.demographic_parity_binary_classification(
            simple_learners.LeastSquaresBinaryClassifierLearner(),
            X, Y, A, number_lagrange_multipliers=7)
        assert len(result) == 7

        result_actual_feature_weights = [
            x["model"].weights["actual_feature"] for x in result]
        result_protected_attribute_feature_weights = [
            x["model"].weights["protected_attribute_feature"] for x in result]
        result_constant_ones_feature_weights = [
            x["model"].weights["constant_ones_feature"] for x in result]

        # The following 'golden values' are empirically determined
        expected_actual_feature_weights = [
            0, 0, 0.22047, 0.95963, 0.68783, 0, 0
        ]

        # Since we have an unfair dataset, a Lagrange multiplier
        # of zero is no longer expected to give a zero weight
        # on the protected attribute
        # However, in the following data, we can see that
        # a non-zero multiplier value should exist to give
        # that zero weight
        expected_protected_attribute_feature_weights = [
            -0.25000,
            -0.25000,
            -0.23338,
            -0.15000,
            0.10659,
            0.25000,
            0.25000]

        expected_constant_ones_feature_weights = [
            1.75,
            1.75,
            1.52549,
            0.77018,
            -0.09694,
            -0.75,
            -0.75]

        assert np.allclose(result_actual_feature_weights,
                           expected_actual_feature_weights,
                           rtol=1e-3)
        assert np.allclose(result_protected_attribute_feature_weights,
                           expected_protected_attribute_feature_weights,
                           rtol=1e-3)
        assert np.allclose(result_constant_ones_feature_weights,
                           expected_constant_ones_feature_weights,
                           rtol=1e-3)
