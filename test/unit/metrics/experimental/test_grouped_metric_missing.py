# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics

from .data_for_test import y_t, y_p, group_gen


def test_missing_sensitive_feature_combinations():
    n = len(y_t)
    g_A = np.asarray([group_gen(x, int(n/2), ['aa', 'bb']) for x in range(n)])
    g_B = np.asarray([group_gen(x, 1+int(n/3), ['x', 'y', 'z']) for x in range(n)])

    target = metrics.GroupedMetric(skm.accuracy_score,
                                   y_t, y_p,
                                   sensitive_features=[g_A, g_B])

    # Make sure our missing combination is in an expected place
    for idx in target.by_group.index:
        mask_A = g_A == idx[0]
        mask_B = g_B == idx[1]
        if idx == ('bb', 'x'):
            assert sum(np.logical_and(mask_A, mask_B)) == 0, 'idx={0}'.format(idx)
        else:
            assert sum(np.logical_and(mask_A, mask_B)) != 0, 'idx={0}'.format(idx)

    assert target.by_group['accuracy_score'][('bb', 'x')] is None
