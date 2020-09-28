# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import functools
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
                                        skm.accuracy_score)

    grouped = metrics.GroupedMetric(skm.accuracy_score,
                                    y_t, y_p,
                                    sensitive_features=gid)

    actual = my_fn(y_t, y_p, sensitive_features=gid)
    assert actual == grouped.difference()['accuracy_score']['overall']


def test_derived_difference_sample_arg():
    my_fbeta = functools.partial(skm.fbeta_score, beta=0.6)
    my_fbeta.__name__ = "my_fbeta"
    my_fn = metrics.make_derived_metric('difference', my_fbeta)

    grouped = metrics.GroupedMetric(my_fbeta,
                                    y_t, y_p,
                                    sensitive_features=gid,
                                    sample_params={'sample_weight': wgt})
    actual = my_fn(y_t, y_p, sensitive_features=gid, sample_weight=wgt)
    assert actual == grouped.difference()['my_fbeta'][0]
