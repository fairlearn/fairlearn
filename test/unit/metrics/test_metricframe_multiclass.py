# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import functools
import numpy as np
import sklearn.metrics as skm

import fairlearn.metrics as metrics

y_true = np.array([0, 1, 2, 0, 1, 2, 2, 1])
y_pred = np.array([
    [0.5, 0.3, 0.2],
    [0.1, 0.5, 0.4],
    [0.1, 0.1, 0.8],
    [0.2, 0.5, 0.3],
    [0.8, 0.1, 0.1],
    [0.7, 0.2, 0.1],
    [0.2, 0.3, 0.5],
    [0.6, 0.1, 0.3]
])

s_f = np.array(['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', ])


def test_roc_auc():
    ras = functools.partial(skm.roc_auc_score,
                            multi_class='ovr',
                            labels=[0, 1, 2])
    target = metrics.MetricFrame(ras,
                                 y_true, y_pred,
                                 sensitive_features=s_f)

    overall = ras(y_true, y_pred)
    assert target.overall == overall

    for g in np.unique(s_f):
        mask = s_f == g
        expected = ras(y_true[mask], y_pred[mask])
        actual = target.by_group[g]
        assert expected == actual
