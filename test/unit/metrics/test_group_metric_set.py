# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

from fairlearn.metrics import group_accuracy_score
from fairlearn.metrics import create_group_metric_set


class TestCreateGroupMetricSet:
    def test_bad_model_type(self):
        with pytest.raises(ValueError) as exception_context:
            create_group_metric_set("Something Random", None, None, None)
        expected = "model_type 'Something Random' not in ['binary_classification', 'regression']"
        assert exception_context.value.args[0] == expected

    def test_smoke(self):
        # Single model, single group vector, no names
        Y_true = [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0]
        Y_pred = [[0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]]
        Groups = [['a', 'b', 'b', 'a', 'b', 'b', 'b', 'a', 'b', 'b', 'b']]
        gr_int = [int(x == 'b') for x in Groups[0]]

        result = create_group_metric_set('binary_classification', Y_true, Y_pred, Groups)
        assert result['predictionType'] == 'binaryClassification'

        assert isinstance(result['trueY'], list)
        assert np.array_equal(result['trueY'], Y_true)

        assert isinstance(result['precomputedFeatureBins'], list)
        assert len(result['precomputedFeatureBins']) == 1
        bin_dict = result['precomputedFeatureBins'][0]
        assert isinstance(bin_dict, dict)
        assert np.array_equal(bin_dict['binVector'], gr_int)
        assert np.array_equal(bin_dict['binLabels'], ['a', 'b'])

        assert isinstance(result['predictedY'], list)
        assert len(result['predictedY']) == 1
        y_p = result['predictedY'][0]
        assert isinstance(y_p, list)
        assert np.array_equal(y_p, Y_pred[0])

        assert isinstance(result['precomputedMetrics'], list)
        assert len(result['precomputedMetrics']) == 1
        metrics_group_0 = result['precomputedMetrics'][0]
        assert isinstance(metrics_group_0, list)
        assert len(metrics_group_0) == 1
        metrics_g0_m0 = metrics_group_0[0]
        assert isinstance(metrics_g0_m0, dict)
        assert len(metrics_g0_m0) == 10
        accuracy = metrics_g0_m0['accuracy_score']
        assert isinstance(accuracy, dict)
        gmr = group_accuracy_score(Y_true, Y_pred[0], Groups[0])
        assert gmr.overall == pytest.approx(accuracy['global'])
        assert isinstance(accuracy['bins'], list)
        assert len(accuracy['bins']) == 2
        assert gmr.by_group['a'] == pytest.approx(accuracy['bins'][0])
        assert gmr.by_group['b'] == pytest.approx(accuracy['bins'][1])

    def test_two_models(self):
        # Two models, single group vector, no names
        Y_true = [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0]
        Y_pred = [[0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                  [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0]]
        Groups = [['a', 'b', 'b', 'a', 'b', 'b', 'b', 'a', 'b', 'b', 'b']]
        gr_int = [int(x == 'b') for x in Groups[0]]

        result = create_group_metric_set('binary_classification', Y_true, Y_pred, Groups)
        assert result['predictionType'] == 'binaryClassification'

        assert isinstance(result['trueY'], list)
        assert np.array_equal(result['trueY'], Y_true)

        assert isinstance(result['precomputedFeatureBins'], list)
        assert len(result['precomputedFeatureBins']) == 1
        bin_dict = result['precomputedFeatureBins'][0]
        assert isinstance(bin_dict, dict)
        assert np.array_equal(bin_dict['binVector'], gr_int)
        assert np.array_equal(bin_dict['binLabels'], ['a', 'b'])

        assert isinstance(result['predictedY'], list)
        assert len(result['predictedY']) == 2
        for i in range(2):
            y_p = result['predictedY'][i]
            assert isinstance(y_p, list)
            assert np.array_equal(y_p, Y_pred[i])

        assert isinstance(result['precomputedMetrics'], list)
        assert len(result['precomputedMetrics']) == 1
        metrics_group_0 = result['precomputedMetrics'][0]
        assert isinstance(metrics_group_0, list)
        assert len(metrics_group_0) == 2
        for i in range(2):
            metrics_g0_m0 = metrics_group_0[i]
            assert isinstance(metrics_g0_m0, dict)
            assert len(metrics_g0_m0) == 10
            accuracy = metrics_g0_m0['accuracy_score']
            assert isinstance(accuracy, dict)
            gmr = group_accuracy_score(Y_true, Y_pred[i], Groups[0])
            assert gmr.overall == pytest.approx(accuracy['global'])
            assert isinstance(accuracy['bins'], list)
            assert len(accuracy['bins']) == 2
            assert gmr.by_group['a'] == pytest.approx(accuracy['bins'][0])
            assert gmr.by_group['b'] == pytest.approx(accuracy['bins'][1])

    def test_two_groups(self):
        # Single model, two group vectors, no names
        Y_true = [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0]
        Y_pred = [[0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]]
        # First group is just 'a' and 'b'. Second is 4, 5 and 6
        Groups = [['a', 'b', 'b', 'a', 'b', 'b', 'b', 'a', 'b', 'b', 'b'],
                  [4, 5, 6, 6, 5, 4, 4, 5, 5, 6, 6]]
        gr_int = [int(x == 'b') for x in Groups[0]]

        result = create_group_metric_set('binary_classification', Y_true, Y_pred, Groups)
        assert result['predictionType'] == 'binaryClassification'

        assert isinstance(result['trueY'], list)
        assert np.array_equal(result['trueY'], Y_true)

        assert isinstance(result['precomputedFeatureBins'], list)
        assert len(result['precomputedFeatureBins']) == 2
        bin_dict0 = result['precomputedFeatureBins'][0]
        assert isinstance(bin_dict0, dict)
        assert np.array_equal(bin_dict0['binVector'], gr_int)
        assert np.array_equal(bin_dict0['binLabels'], ['a', 'b'])
        bin_dict1 = result['precomputedFeatureBins'][1]
        assert isinstance(bin_dict1, dict)
        assert np.array_equal(bin_dict1['binVector'], [x-4 for x in Groups[1]])
        assert np.array_equal(bin_dict1['binLabels'], ['4', '5', '6'])

        assert isinstance(result['predictedY'], list)
        assert len(result['predictedY']) == 1
        y_p = result['predictedY'][0]
        assert isinstance(y_p, list)
        assert np.array_equal(y_p, Y_pred[0])

        assert isinstance(result['precomputedMetrics'], list)
        assert len(result['precomputedMetrics']) == 2

        # Check the first grouping (with alphabetical labels)
        metrics_group_0 = result['precomputedMetrics'][0]
        assert isinstance(metrics_group_0, list)
        assert len(metrics_group_0) == 1
        metrics_g0_m0 = metrics_group_0[0]
        assert isinstance(metrics_g0_m0, dict)
        assert len(metrics_g0_m0) == 10
        accuracy = metrics_g0_m0['accuracy_score']
        assert isinstance(accuracy, dict)
        gmr = group_accuracy_score(Y_true, Y_pred[0], Groups[0])
        assert gmr.overall == pytest.approx(accuracy['global'])
        assert isinstance(accuracy['bins'], list)
        assert len(accuracy['bins']) == 2
        assert gmr.by_group['a'] == pytest.approx(accuracy['bins'][0])
        assert gmr.by_group['b'] == pytest.approx(accuracy['bins'][1])

        # Check the second grouping (three unique numeric labels)
        metrics_group_1 = result['precomputedMetrics'][1]
        assert isinstance(metrics_group_1, list)
        assert len(metrics_group_1) == 1
        metrics_g1_m0 = metrics_group_1[0]
        assert isinstance(metrics_g1_m0, dict)
        assert len(metrics_g1_m0) == 10
        accuracy = metrics_g1_m0['accuracy_score']
        assert isinstance(accuracy, dict)
        gmr = group_accuracy_score(Y_true, Y_pred[0], Groups[1])
        assert gmr.overall == pytest.approx(accuracy['global'])
        assert isinstance(accuracy['bins'], list)
        assert len(accuracy['bins']) == 3
        for i in range(3):
            assert gmr.by_group[i+4] == pytest.approx(accuracy['bins'][i])

    def test_two_named_groups(self):
        # Single model, two group vectors, no names
        Y_true = [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0]
        Y_pred = [[0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]]
        # First group is just 'a' and 'b'. Second is 4, 5 and 6
        Groups = [['a', 'b', 'b', 'a', 'b', 'b', 'b', 'a', 'b', 'b', 'b'],
                  [4, 5, 6, 6, 5, 4, 4, 5, 5, 6, 6]]
        gr_int = [int(x == 'b') for x in Groups[0]]
        group_titles = ['alpha', 'num']

        result = create_group_metric_set('binary_classification',
                                         Y_true, Y_pred, Groups,
                                         group_titles=group_titles)
        assert result['predictionType'] == 'binaryClassification'

        assert isinstance(result['trueY'], list)
        assert np.array_equal(result['trueY'], Y_true)

        assert isinstance(result['precomputedFeatureBins'], list)
        assert len(result['precomputedFeatureBins']) == 2
        bin_dict0 = result['precomputedFeatureBins'][0]
        assert isinstance(bin_dict0, dict)
        assert np.array_equal(bin_dict0['binVector'], gr_int)
        assert np.array_equal(bin_dict0['binLabels'], ['a', 'b'])
        assert group_titles[0] == bin_dict0['featureBinName']
        bin_dict1 = result['precomputedFeatureBins'][1]
        assert isinstance(bin_dict1, dict)
        assert np.array_equal(bin_dict1['binVector'], [x-4 for x in Groups[1]])
        assert np.array_equal(bin_dict1['binLabels'], ['4', '5', '6'])
        assert group_titles[1] == bin_dict1['featureBinName']

    def test_two_named_models(self):
        # Two models, single group vector, no names
        Y_true = [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0]
        Y_pred = [[0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                  [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0]]
        Groups = [['a', 'b', 'b', 'a', 'b', 'b', 'b', 'a', 'b', 'b', 'b']]
        gr_int = [int(x == 'b') for x in Groups[0]]
        model_names = ['firstModel', 'secondModel']

        result = create_group_metric_set('binary_classification',
                                         Y_true, Y_pred, Groups,
                                         model_titles=model_names)
        assert result['predictionType'] == 'binaryClassification'

        assert isinstance(result['trueY'], list)
        assert np.array_equal(result['trueY'], Y_true)

        assert isinstance(result['precomputedFeatureBins'], list)
        assert len(result['precomputedFeatureBins']) == 1
        bin_dict = result['precomputedFeatureBins'][0]
        assert isinstance(bin_dict, dict)
        assert np.array_equal(bin_dict['binVector'], gr_int)
        assert np.array_equal(bin_dict['binLabels'], ['a', 'b'])

        assert isinstance(result['modelNames'], list)
        assert np.array_equal(result['modelNames'], model_names)
