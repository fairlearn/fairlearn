# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pytest

import sklearn.metrics as skm

from fairlearn.metrics import MetricFrame
from fairlearn.metrics._group_metric_set import _process_predictions
from fairlearn.metrics._group_metric_set import _process_sensitive_features
from fairlearn.metrics._group_metric_set import _create_group_metric_set

from .sample_loader import load_sample_dashboard
from test.unit.input_convertors import conversions_for_1d

_BC_1P_1F = "bc-1p-1f.json"
_BC_2P_3F = "bc-2p-3f.json"


def validate_dashboard_dictionary(dashboard_dict):
    """Ensure dictionary is a valid Dashboard."""
    schema_type = dashboard_dict['schemaType']
    assert schema_type == 'dashboardDictionary'
    schema_version = dashboard_dict['schemaVersion']
    # Will want to update the following prior to release
    assert schema_version == 0

    pred_type = dashboard_dict['predictionType']
    assert pred_type in {'binaryClassification', 'regression', 'probability'}
    len_y_true = len(dashboard_dict['trueY'])
    num_y_pred = len(dashboard_dict['predictedY'])
    for y_pred in dashboard_dict['predictedY']:
        assert len(y_pred) == len_y_true

    len_model_names = len(dashboard_dict['modelNames'])
    assert len_model_names == num_y_pred

    num_sf = len(dashboard_dict['precomputedFeatureBins'])
    for sf in dashboard_dict['precomputedFeatureBins']:
        sf_vector = sf['binVector']
        assert len(sf_vector) == len_y_true
        for val in sf_vector:
            assert isinstance(val, int)
        sf_classes = sf['binLabels']
        assert len(sf_classes) == 1 + max(sf_vector)

    expected_keys = sorted(list(dashboard_dict['precomputedMetrics'][0][0].keys()))
    assert len(dashboard_dict['precomputedMetrics']) == num_sf
    for metrics_arr in dashboard_dict['precomputedMetrics']:
        assert len(metrics_arr) == num_y_pred
        for m in metrics_arr:
            keys = sorted(list(m.keys()))
            assert keys == expected_keys


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

    @pytest.mark.parametrize("transform_feature", conversions_for_1d)
    def test_smoke_string_groups(self, transform_feature):
        sf_name = "My SF"
        sf_vals = transform_feature(['b', 'a', 'c', 'a', 'b'])

        sf = {sf_name: sf_vals}
        result = _process_sensitive_features(sf)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]['featureBinName'] == sf_name
        assert result[0]['binVector'] == [1, 0, 2, 0, 1]
        assert result[0]['binLabels'] == ["a", "b", "c"]

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
    @pytest.mark.parametrize("t_y_t", conversions_for_1d)
    @pytest.mark.parametrize("t_y_p", conversions_for_1d)
    @pytest.mark.parametrize("t_sf", conversions_for_1d)
    def test_round_trip_1p_1f(self, t_y_t, t_y_p, t_sf):
        expected = load_sample_dashboard(_BC_1P_1F)

        y_true = t_y_t(expected['trueY'])
        y_pred = {expected['modelNames'][0]: t_y_p(expected['predictedY'][0])}

        sf_file = expected['precomputedFeatureBins'][0]
        sf = [sf_file['binLabels'][x] for x in sf_file['binVector']]
        sensitive_feature = {sf_file['featureBinName']: t_sf(sf)}

        actual = _create_group_metric_set(y_true,
                                          y_pred,
                                          sensitive_feature,
                                          'binary_classification')
        validate_dashboard_dictionary(actual)
        assert expected == actual

    @pytest.mark.parametrize("t_y_t", conversions_for_1d)
    @pytest.mark.parametrize("t_y_p", conversions_for_1d)
    @pytest.mark.parametrize("t_sf", conversions_for_1d)
    def test_round_trip_2p_3f(self, t_y_t, t_y_p, t_sf):
        expected = load_sample_dashboard(_BC_2P_3F)

        y_true = t_y_t(expected['trueY'])

        y_pred = {}
        y_p_ts = [t_y_p, lambda x: x]  # Only transform one y_p
        for i, name in enumerate(expected['modelNames']):
            y_pred[name] = y_p_ts[i](expected['predictedY'][i])

        sensitive_features = {}
        t_sfs = [lambda x: x, t_sf, lambda x: x]  # Only transform one sf
        for i, sf_file in enumerate(expected['precomputedFeatureBins']):
            sf = [sf_file['binLabels'][x] for x in sf_file['binVector']]
            sensitive_features[sf_file['featureBinName']] = t_sfs[i](sf)

        actual = _create_group_metric_set(y_true,
                                          y_pred,
                                          sensitive_features,
                                          'binary_classification')
        validate_dashboard_dictionary(actual)
        assert expected == actual

    def test_specific_metrics(self):
        y_t = [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1]
        y_p = [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0]
        s_f = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1]

        expected = MetricFrame({'accuracy_score': skm.accuracy_score,
                                'roc_auc_score': skm.roc_auc_score},
                               y_t, y_p,
                               sensitive_features=s_f)

        predictions = {"some model": y_p}
        sensitive_feature = {"my sf": s_f}

        actual = _create_group_metric_set(y_t,
                                          predictions,
                                          sensitive_feature,
                                          'binary_classification')

        # Do some sanity checks
        validate_dashboard_dictionary(actual)
        assert actual['trueY'] == y_t
        assert actual['predictedY'][0] == y_p
        assert actual['precomputedFeatureBins'][0]['binVector'] == s_f
        assert len(actual['precomputedMetrics'][0][0]) == 11

        # Cross check the two metrics we computed
        # Comparisons simplified because s_f was already {0,1}
        actual_acc = actual['precomputedMetrics'][0][0]['accuracy_score']
        assert actual_acc['global'] == expected.overall['accuracy_score']
        assert actual_acc['bins'] == list(expected.by_group['accuracy_score'])

        actual_roc = actual['precomputedMetrics'][0][0]['balanced_accuracy_score']
        assert actual_roc['global'] == expected.overall['roc_auc_score']
        assert actual_roc['bins'] == list(expected.by_group['roc_auc_score'])

    def test_regression_prediction_type(self):
        y_t = [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1]
        y_p = [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0]
        s_f = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1]

        predictions = {"some model": y_p}
        sensitive_feature = {"my sf": s_f}

        # Using the `regression` prediction type should not crash
        _create_group_metric_set(y_t,
                                 predictions,
                                 sensitive_feature,
                                 'regression')
