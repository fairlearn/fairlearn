# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

from fairlearn.metrics import group_accuracy_score, group_roc_auc_score
from fairlearn.metrics._group_metric_set import _process_feature_to_integers
from fairlearn.metrics._group_metric_set import _process_predictions
from fairlearn.metrics._group_metric_set import _process_sensitive_features
from fairlearn.metrics._group_metric_set import create_group_metric_set

from .sample_loader import load_sample_dashboard
from test.unit.input_convertors import conversions_for_1d

_BC_1P_1F = "bc-1p-1f.json"
_BC_2P_3F = "bc-2p-3f.json"


class TestProcessFeatureToInteger:
    @pytest.mark.parametrize("transform_feature", conversions_for_1d)
    def test_smoke(self, transform_feature):
        f = transform_feature([1, 2, 1, 2])

        names, f_integer = _process_feature_to_integers(f)
        assert isinstance(names, list)
        assert isinstance(f_integer, list)
        assert np.array_equal(names, ["1", "2"])
        assert np.array_equal(f_integer, [0, 1, 0, 1])

    @pytest.mark.parametrize("transform_feature", conversions_for_1d)
    def test_strings(self, transform_feature):
        f = transform_feature(['b', 'a', 'a', 'b'])

        names, f_integer = _process_feature_to_integers(f)
        assert isinstance(names, list)
        assert isinstance(f_integer, list)
        assert np.array_equal(names, ["a", "b"])
        assert np.array_equal(f_integer, [1, 0, 0, 1])


class TestProcessSensitiveFeatures:
    @pytest.mark.parametrize("transform_feature", conversions_for_1d)
    def test_smoke(self, transform_feature):
        sf_name = "My SF"
        sf_vals = transform_feature([1, 3, 3, 1])

        sf = {sf_name: sf_vals}
        result = _process_sensitive_features(sf)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]['featureBinName'] == sf_name
        assert result[0]['binVector'] == [0, 1, 1, 0]
        assert result[0]['binLabels'] == ["1", "3"]

    def test_result_is_sorted(self):
        sf_vals = [1, 2, 3, 1]

        sf = {"b": sf_vals, "a": sf_vals, "c": sf_vals}
        result = _process_sensitive_features(sf)
        assert isinstance(result, list)
        assert len(result) == 3
        for r in result:
            assert r['binVector'] == [0, 1, 2, 0]
            assert r['binLabels'] == ['1', '2', '3']
        result_names = [r['featureBinName'] for r in result]
        assert result_names == ["a", "b", "c"]


class TestProcessPredictions:
    @pytest.mark.parametrize("transform_y_p", conversions_for_1d)
    def test_smoke(self, transform_y_p):
        y_pred = transform_y_p([0, 1, 1, 0])
        name = "my model"

        predictions = {name: y_pred}
        names, preds = _process_predictions(predictions)
        assert isinstance(names, list)
        assert isinstance(preds, list)
        assert len(names) == 1
        assert len(preds) == 1
        assert names[0] == name
        assert isinstance(preds[0], list)
        assert preds[0] == [0, 1, 1, 0]

    @pytest.mark.parametrize("transform_y_1", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_2", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_3", conversions_for_1d)
    def test_results_are_sorted(self,
                                transform_y_1,
                                transform_y_2,
                                transform_y_3):
        y_p1 = transform_y_1([0, 0, 1, 1])
        y_p2 = transform_y_2([0, 1, 0, 1])
        y_p3 = transform_y_3([1, 1, 0, 0])
        predictions = {"b": y_p1, "a": y_p2, "c": y_p3}

        names, preds = _process_predictions(predictions)
        assert names == ["a", "b", "c"]
        for i in range(3):
            assert isinstance(preds[i], list)
        assert preds[0] == [0, 1, 0, 1]
        assert preds[1] == [0, 0, 1, 1]
        assert preds[2] == [1, 1, 0, 0]


class TestCreateGroupMetricSet:
    def test_round_trip_1p_1f(self):
        expected = load_sample_dashboard(_BC_1P_1F)

        y_true = expected['trueY']
        y_pred = {expected['modelNames'][0]: expected['predictedY'][0]}

        sf_file = expected['precomputedFeatureBins'][0]
        sf = [sf_file['binLabels'][x] for x in sf_file['binVector']]
        sensitive_feature = {sf_file['featureBinName']: sf}

        actual = create_group_metric_set(y_true,
                                         y_pred,
                                         sensitive_feature,
                                         'binary_classification')
        assert expected == actual
