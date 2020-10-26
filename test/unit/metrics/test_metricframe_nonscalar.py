# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import functools
import numpy as np
import sklearn.metrics as skm

import fairlearn.metrics as metrics

from .data_for_test import y_t, y_p, g_1, g_2


def test_1m_1sf_0cf():
    target = metrics.MetricFrame({'confusion_matrix': skm.confusion_matrix},
                                 y_t,
                                 y_p,
                                 sensitive_features=g_1)

    overall = skm.confusion_matrix(y_t, y_p)
    assert np.array_equal(target.overall['confusion_matrix'], overall)

    for g in np.unique(g_1):
        mask = g_1 == g
        expected = skm.confusion_matrix(y_t[mask], y_p[mask])
        actual = target.by_group['confusion_matrix'][g]
        assert np.array_equal(actual, expected)


def test_1m_1sf_1cf():
    target = metrics.MetricFrame(skm.confusion_matrix,
                                 y_t,
                                 y_p,
                                 sensitive_features=g_1,
                                 control_features=g_2)

    for cf in np.unique(g_2):
        mask_c = g_2 == cf
        overall = skm.confusion_matrix(y_t[mask_c], y_p[mask_c])
        actual = target.overall[cf]
        assert np.array_equal(overall, actual)

        for sf in np.unique(g_1):
            mask_s = g_1 == sf
            mask = np.logical_and(mask_c, mask_s)
            expected = skm.confusion_matrix(y_t[mask], y_p[mask])
            actual = target.by_group[(cf, sf)]
            assert np.array_equal(expected, actual)


def test_mixed_metrics():
    metric_dict = {'cm': skm.confusion_matrix, 'prec': skm.precision_score}

    target = metrics.MetricFrame(metric_dict,
                                 y_t, y_p,
                                 sensitive_features=g_1)

    assert target.overall['prec'] == skm.precision_score(y_t, y_p)
    assert np.array_equal(target.overall['cm'], skm.confusion_matrix(y_t, y_p))

    for g in np.unique(g_1):
        mask = g_1 == g

        expected = skm.confusion_matrix(y_t[mask], y_p[mask])
        actual = target.by_group['cm'][g]
        assert np.array_equal(actual, expected)

        expected = skm.precision_score(y_t[mask], y_p[mask])
        actual = target.by_group['prec'][g]
        assert expected == actual


def test_multid_input_output():
    # In this, both y_t and y_p are 2d arrays
    # The metric results are also arrays
    metric_fn = functools.partial(skm.r2_score, multioutput='raw_values')

    y_t_2 = np.random.rand(len(g_1), 2)
    y_p_2 = np.random.rand(len(g_1), 2)

    target = metrics.MetricFrame(metric_fn, y_t_2, y_p_2, sensitive_features=g_1)

    expected_overall = skm.r2_score(y_t_2, y_p_2, multioutput='raw_values')
    assert np.array_equal(target.overall, expected_overall)
    for g in np.unique(g_1):
        mask = g_1 == g

        expected = skm.r2_score(y_t_2[mask], y_p_2[mask], multioutput='raw_values')
        actual = target.by_group[g]
        assert np.array_equal(actual, expected)
