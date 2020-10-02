# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics

from .data_for_test import y_t, y_p, g_1, g_2


def test_1m_1sf_0cf():
    target = metrics.GroupedMetric(skm.confusion_matrix,
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
    target = metrics.GroupedMetric(skm.confusion_matrix,
                                   y_t,
                                   y_p,
                                   sensitive_features=g_1,
                                   conditional_features=g_2)

    for cf in np.unique(g_2):
        mask_c = g_2 == cf
        overall = skm.confusion_matrix(y_t[mask_c], y_p[mask_c])
        actual = target.overall['confusion_matrix'][cf]
        assert np.array_equal(overall, actual)

        for sf in np.unique(g_1):
            mask_s = g_1 == sf
            mask = np.logical_and(mask_c, mask_s)
            expected = skm.confusion_matrix(y_t[mask], y_p[mask])
            actual = target.by_group['confusion_matrix'][(cf, sf)]
            assert np.array_equal(expected, actual)


def test_mixed_metrics():
    metric_dict = {'cm': skm.confusion_matrix, 'prec': skm.precision_score}

    target = metrics.GroupedMetric(metric_dict,
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
