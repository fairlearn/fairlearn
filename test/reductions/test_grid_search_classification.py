# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from fairlearn.metrics.demographic_parity import DemographicParity
from fairlearn.reductions.grid_search_classification import GridSearchClassification

import numpy as np
import pandas as pd

from test import simple_learners


class TestGridSearchClassification:
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

    def test_demographicparity_fair_uneven_populations(self):
        # Variant of test_demographicparity_already_fair, which has unequal
        # populations in the two classes
        # Also allow the threshold to be adjustable

        score_threshold = 0.625

        number_a0 = 5000
        number_a1 = 200

        a0_label = 17
        a1_label = 37

        X, Y, A = self._simple_threshold_data(number_a0, number_a1,
                                              score_threshold, score_threshold,
                                              a0_label, a1_label)

        target = GridSearchClassification(simple_learners.LeastSquaresBinaryClassifierLearner(),
                                          fairness_metric=DemographicParity(),
                                          number_of_lagrange_multipliers=11)

        target.fit(X, Y, A)

        assert len(target.all_results) == 11

        result_actual_feature_weights = [
            x["model"].weights["actual_feature"] for x in target.all_results]
        result_protected_attribute_feature_weights = [
            x["model"].weights["protected_attribute_feature"] for x in target.all_results]
        result_constant_ones_feature_weights = [
            x["model"].weights["constant_ones_feature"] for x in target.all_results]

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
