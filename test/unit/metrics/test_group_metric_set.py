# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

from fairlearn.metrics import group_accuracy_score, group_balanced_root_mean_squared_error
from fairlearn.metrics import GroupMetricSet, GroupMetricResult
from fairlearn.metrics._group_metric_set import create_group_metric_set


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


class TestProperties:
    def test_model_type_property(self):
        target = GroupMetricSet()

        for mt in GroupMetricSet._allowed_model_types:
            target.model_type = mt
            assert target.model_type == mt

    def test_model_type_not_allowed(self):
        target = GroupMetricSet()

        with pytest.raises(ValueError) as exception_context:
            target.model_type = "Something Random"
        expected = "model_type 'Something Random' not in ['binary_classification', 'regression']"
        assert exception_context.value.args[0] == expected

    def test_y_true(self):
        target = GroupMetricSet()
        target.y_true = [1, 2, 3]
        assert isinstance(target.y_true, np.ndarray)
        assert np.array_equal(target.y_true, [1, 2, 3])

    def test_y_pred(self):
        target = GroupMetricSet()
        target.y_pred = [1, 2]
        assert isinstance(target.y_pred, np.ndarray)
        assert np.array_equal(target.y_pred, [1, 2])

    def test_groups(self):
        target = GroupMetricSet()
        target.groups = [0, 1, 2]
        assert isinstance(target.groups, np.ndarray)
        assert np.array_equal(target.groups, [0, 1, 2])

    def test_groups_not_from_zero(self):
        target = GroupMetricSet()
        with pytest.raises(ValueError) as exception_context:
            target.groups = [4, 5, 6]
        msg = "The unique values of the groups property must be sequential integers from zero"
        assert exception_context.value.args[0] == msg

    def test_groups_not_sequential(self):
        target = GroupMetricSet()
        with pytest.raises(ValueError) as exception_context:
            target.groups = [0, 2, 4]
        msg = "The unique values of the groups property must be sequential integers from zero"
        assert exception_context.value.args[0] == msg

    def test_groups_strings(self):
        target = GroupMetricSet()
        with pytest.raises(ValueError) as exception_context:
            target.groups = ['0', '1', '2']
        msg = "The unique values of the groups property must be sequential integers from zero"
        assert exception_context.value.args[0] == msg

    def test_group_names(self):
        target = GroupMetricSet()
        target.group_names = ['a', 'b']
        assert target.group_names[0] == 'a'
        assert target.group_names[1] == 'b'

    def test_group_names_not_list(self):
        target = GroupMetricSet()
        with pytest.raises(ValueError) as exception_context:
            target.group_names = {'a': 'b', 'c': 'd'}
        expected = "The group_names property must be a list of strings"
        assert exception_context.value.args[0] == expected

    def test_group_names_values_not_string(self):
        target = GroupMetricSet()
        with pytest.raises(ValueError) as exception_context:
            target.group_names = [0, 1, 2, 'a']
        expected = "The group_names property must be a list of strings"
        assert exception_context.value.args[0] == expected

    def test_metrics(self):
        target = GroupMetricSet()
        my_metric = GroupMetricResult()
        my_metric.overall = 10222
        target.metrics = {"60bdb14d-83fb-4374-ab2e-4c371f22b21e": my_metric}
        assert target.metrics["60bdb14d-83fb-4374-ab2e-4c371f22b21e"].overall == 10222

    def test_metrics_keys_not_string(self):
        target = GroupMetricSet()
        my_metric = GroupMetricResult()
        my_metric.overall = 10222
        with pytest.raises(ValueError) as exception_context:
            target.metrics = {0: my_metric}
        expected = "Keys for metrics dictionary must be strings"
        assert exception_context.value.args[0] == expected

    def test_metrics_values_not_groupmetricresult(self):
        target = GroupMetricSet()
        with pytest.raises(ValueError) as exception_context:
            target.metrics = {"a": 0}
        expected = "Values for metrics dictionary must be of type GroupMetricResult"
        assert exception_context.value.args[0] == expected


class TestConsistencyCheck:
    def test_length_mismatch_y_true(self):
        target = GroupMetricSet()
        target.y_true = [0, 1, 0]
        target.y_pred = [0, 1, 1, 1]
        target.groups = [0, 1, 1, 1]

        with pytest.raises(ValueError) as exception_context:
            target.check_consistency()
        assert exception_context.value.args[0] == "Lengths of y_true, y_pred and groups must match"

    def test_length_mismatch_y_pred(self):
        target = GroupMetricSet()
        target.y_true = [0, 1, 0, 1]
        target.y_pred = [0, 1, 1]
        target.groups = [0, 1, 1, 1]

        with pytest.raises(ValueError) as exception_context:
            target.check_consistency()
        assert exception_context.value.args[0] == "Lengths of y_true, y_pred and groups must match"

    def test_length_mismatch_groups(self):
        target = GroupMetricSet()
        target.y_true = [0, 1, 0, 1]
        target.y_pred = [0, 1, 1, 0]
        target.groups = [0, 1, 1]

        with pytest.raises(ValueError) as exception_context:
            target.check_consistency()
        assert exception_context.value.args[0] == "Lengths of y_true, y_pred and groups must match"

    def test_metric_has_bad_groups(self):
        target = GroupMetricSet()
        target.y_true = [0, 1, 1, 1, 0]
        target.y_pred = [1, 1, 1, 0, 0]
        target.groups = [0, 1, 0, 1, 1]
        bad_metric = GroupMetricResult()
        bad_metric.by_group[0] = 0.1
        metric_dict = {'bad_metric': bad_metric}
        target.metrics = metric_dict

        with pytest.raises(ValueError) as exception_context:
            target.check_consistency()
        expected = "The groups for metric bad_metric do not match the groups property"
        assert exception_context.value.args[0] == expected

    def test_group_names_do_not_match_groups(self):
        target = GroupMetricSet()

        target.model_type = GroupMetricSet.BINARY_CLASSIFICATION
        target.y_true = [0, 1, 0, 0]
        target.y_pred = [1, 1, 1, 0]
        target.groups = [0, 1, 1, 0]

        # Some wholly synthetic metrics
        firstMetric = GroupMetricResult()
        firstMetric.overall = 0.2
        firstMetric.by_group[0] = 0.3
        firstMetric.by_group[1] = 0.4
        secondMetric = GroupMetricResult()
        secondMetric.overall = 0.6
        secondMetric.by_group[0] = 0.7
        secondMetric.by_group[1] = 0.8
        metric_dict = {GroupMetricSet.GROUP_ACCURACY_SCORE: firstMetric,
                       GroupMetricSet.GROUP_MISS_RATE: secondMetric}

        target.metrics = metric_dict

        target.group_names = ['First']
        target.group_title = "Some string"
        with pytest.raises(ValueError) as exception_context:
            target.check_consistency()
        expected = "Count of group_names not the same as the number of unique groups"
        assert exception_context.value.args[0] == expected


class TestDictionaryConversions:
    def test_to_dict_smoke(self):
        target = GroupMetricSet()

        target.model_type = GroupMetricSet.BINARY_CLASSIFICATION
        target.y_true = [0, 1, 0, 0]
        target.y_pred = [1, 1, 1, 0]
        target.groups = [0, 1, 1, 0]

        # Some wholly synthetic metrics
        firstMetric = GroupMetricResult()
        firstMetric.overall = 0.2
        firstMetric.by_group[0] = 0.3
        firstMetric.by_group[1] = 0.4
        secondMetric = GroupMetricResult()
        secondMetric.overall = 0.6
        secondMetric.by_group[0] = 0.7
        secondMetric.by_group[1] = 0.8
        metric_dict = {GroupMetricSet.GROUP_ACCURACY_SCORE: firstMetric,
                       GroupMetricSet.GROUP_MISS_RATE: secondMetric}

        target.metrics = metric_dict

        target.group_names = ['First', 'Second']
        target.group_title = "Some string"

        result = target.to_dict()

        assert result['predictionType'] == 'binaryClassification'
        assert np.array_equal(target.y_true, result['trueY'])
        assert len(result['predictedY']) == 1
        assert np.array_equal(result['predictedY'][0], target.y_pred)

        assert len(result['precomputedMetrics']) == 1
        assert len(result['precomputedMetrics'][0]) == 1
        rmd = result['precomputedMetrics'][0][0]
        assert len(rmd) == 2
        assert rmd['accuracy_score']['global'] == 0.2
        assert rmd['accuracy_score']['bins'][0] == 0.3
        assert rmd['accuracy_score']['bins'][1] == 0.4
        assert rmd['miss_rate']['global'] == 0.6
        assert rmd['miss_rate']['bins'][0] == 0.7
        assert rmd['miss_rate']['bins'][1] == 0.8
        assert result['precomputedFeatureBins'][0]['featureBinName'] == "Some string"
        assert np.array_equal(result['precomputedFeatureBins'][0]['binLabels'],
                              ['First', 'Second'])

    def test_round_trip_smoke(self):
        original = GroupMetricSet()

        original.model_type = GroupMetricSet.BINARY_CLASSIFICATION
        original.y_true = [0, 1, 0, 0]
        original.y_pred = [1, 1, 1, 0]
        original.groups = [0, 1, 2, 0]
        original.group_title = 123

        # Some wholly synthetic metrics
        firstMetric = GroupMetricResult()
        firstMetric.overall = 0.2
        firstMetric.by_group[0] = 0.25
        firstMetric.by_group[1] = 0.5
        firstMetric.by_group[2] = 0.2
        secondMetric = GroupMetricResult()
        secondMetric.overall = 0.6
        secondMetric.by_group[0] = 0.75
        secondMetric.by_group[1] = 0.25
        secondMetric.by_group[2] = 0.25
        metric_dict = {GroupMetricSet.GROUP_ACCURACY_SCORE: firstMetric,
                       GroupMetricSet.GROUP_MISS_RATE: secondMetric}
        original.metrics = metric_dict
        original.group_names = ['First', 'Second', 'Something else']

        intermediate_dict = original.to_dict()

        result = GroupMetricSet.from_dict(intermediate_dict)

        assert original == result


Y_true = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
Y_pred = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
groups = [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0]
gr_in2 = [2*(x+1) for x in groups]
gr_alp = ['a' if x == 0 else 'b' for x in groups]


def test_compute_binary():
    target = GroupMetricSet()

    target.compute(Y_true, Y_pred, groups, model_type=GroupMetricSet.BINARY_CLASSIFICATION)

    sample_expected = group_accuracy_score(Y_true, Y_pred, groups)

    assert np.array_equal(Y_true, target.y_true)
    assert np.array_equal(Y_pred, target.y_pred)
    assert np.array_equal(groups, target.groups)
    assert np.array_equal(['0', '1'], target.group_names)
    assert len(target.metrics) == 10
    assert target.metrics[GroupMetricSet.GROUP_ACCURACY_SCORE].overall == sample_expected.overall
    for g in np.unique(groups):
        assert (target.metrics[GroupMetricSet.GROUP_ACCURACY_SCORE].by_group[g] ==
                sample_expected.by_group[g])


def test_compute_regression():
    target = GroupMetricSet()

    target.compute(Y_true, Y_pred, groups, model_type=GroupMetricSet.REGRESSION)

    sample_expected = group_balanced_root_mean_squared_error(Y_true, Y_pred, groups)

    assert np.array_equal(Y_true, target.y_true)
    assert np.array_equal(Y_pred, target.y_pred)
    assert np.array_equal(groups, target.groups)
    assert np.array_equal(['0', '1'], target.group_names)
    assert len(target.metrics) == 12
    assert target.metrics[GroupMetricSet.GROUP_BALANCED_ROOT_MEAN_SQUARED_ERROR].overall == sample_expected.overall  # noqa: E501
    for g in np.unique(groups):
        assert (target.metrics[GroupMetricSet.GROUP_BALANCED_ROOT_MEAN_SQUARED_ERROR].by_group[g]
                == sample_expected.by_group[g])


def test_groups_not_sequential_int():
    target = GroupMetricSet()
    regular = GroupMetricSet()

    # Make 'target' the same as 'regular' but with different integers for the groups
    target.compute(Y_true, Y_pred, gr_in2, model_type=GroupMetricSet.BINARY_CLASSIFICATION)
    regular.compute(Y_true, Y_pred, groups, model_type=GroupMetricSet.BINARY_CLASSIFICATION)

    assert np.array_equal(['2', '4'], target.group_names)

    assert target.metrics == regular.metrics


def test_groups_alphabetical():
    target = GroupMetricSet()
    regular = GroupMetricSet()

    # Make 'target' the same as 'regular' but with strings for the groups
    target.compute(Y_true, Y_pred, gr_alp, model_type=GroupMetricSet.BINARY_CLASSIFICATION)
    regular.compute(Y_true, Y_pred, groups, model_type=GroupMetricSet.BINARY_CLASSIFICATION)

    assert np.array_equal(['a', 'b'], target.group_names)

    tm = target.metrics[GroupMetricSet.GROUP_ACCURACY_SCORE]
    rm = target.metrics[GroupMetricSet.GROUP_ACCURACY_SCORE]

    assert tm.overall == rm.overall
    assert tm.by_group == rm.by_group
    assert tm == rm

    assert target.metrics == regular.metrics


def test_equality():
    a = GroupMetricSet()
    b = GroupMetricSet()

    a.compute(Y_true, Y_pred, gr_alp, model_type=GroupMetricSet.BINARY_CLASSIFICATION)
    b.compute(Y_true, Y_pred, gr_alp, model_type=GroupMetricSet.BINARY_CLASSIFICATION)

    assert a == b
    assert not(a != b)


def test_inequality():
    a = GroupMetricSet()
    b = GroupMetricSet()

    a.compute(Y_true, Y_pred, groups, model_type=GroupMetricSet.BINARY_CLASSIFICATION)
    b.compute(Y_true, Y_pred, gr_alp, model_type=GroupMetricSet.BINARY_CLASSIFICATION)

    assert not(a == b)
    assert a != b
