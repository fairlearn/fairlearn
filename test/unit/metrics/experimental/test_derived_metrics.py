# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics


# ============================================================

y_t_A = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
y_p_A = [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
wgt_A = [1, 2, 1, 2, 1, 2, 1, 3, 4, 1, 2, 1, 2, 1, 2, 1, 2, 5]
As = np.full(len(y_t_A), 'A')
y_t_B = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
y_p_B = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1]
wgt_B = [2, 2, 1, 1, 3, 3, 4, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1]
Bs = np.full(len(y_t_B), 'B')

y_t = np.concatenate((y_t_A, y_t_B))
y_p = np.concatenate((y_p_A, y_p_B))
wgt = np.concatenate((wgt_A, wgt_B))
gid = np.concatenate((As, Bs))


def test_derived_difference():
    my_fn = metrics.make_derived_metric('difference',
                                        skm.accuracy_score,
                                        sample_param_names=['sample_weight'])

    grouped = metrics.GroupedMetric(skm.accuracy_score,
                                    y_t, y_p,
                                    sensitive_features=gid)

    actual = my_fn(y_t, y_p, sensitive_features=gid)
    assert actual == grouped.difference()['accuracy_score']['overall']


def test_derived_difference_scalar_arg():
    my_fn = metrics.make_derived_metric('difference',
                                        skm.fbeta_score,
                                        sample_param_names=['sample_weight'])
    grouped = metrics.GroupedMetric(skm.fbeta_score,
                                    y_t, y_p,
                                    sensitive_features=gid,
                                    params={'beta': 0.6})
    actual = my_fn(y_t, y_p, sensitive_features=gid, beta=0.6)
    assert actual == grouped.difference()['fbeta_score'][0]


def test_derived_difference_both_arg_types():
    my_fn = metrics.make_derived_metric('difference',
                                        skm.fbeta_score,
                                        sample_param_names=['sample_weight'])
    grouped = metrics.GroupedMetric(skm.fbeta_score,
                                    y_t, y_p,
                                    sensitive_features=gid,
                                    sample_param_names=['sample_weight'],
                                    params={'beta': 0.6, 'sample_weight': wgt})
    actual = my_fn(y_t, y_p, sensitive_features=gid, beta=0.6, sample_weight=wgt)
    assert actual == grouped.difference()['fbeta_score'][0]
