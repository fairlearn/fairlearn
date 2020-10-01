# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics

from .data_for_test import y_t, y_p, group_gen

metric_functions = [skm.recall_score,
                    skm.precision_score,
                    skm.accuracy_score,
                    skm.balanced_accuracy_score]


@pytest.mark.parametrize("metric_fn", metric_functions)
def test_missing_sensitive_feature_combinations(metric_fn):
    n = len(y_t)
    g_A = np.asarray([group_gen(x, int(n/2), ['aa', 'bb']) for x in range(n)])
    g_B = np.asarray([group_gen(x, 1+int(n/3), ['x', 'y', 'z']) for x in range(n)])

    target = metrics.GroupedMetric(metric_fn,
                                   y_t, y_p,
                                   sensitive_features=[g_A, g_B])

    # Make sure our missing combination is in an expected place
    overall = metric_fn(y_t, y_p)
    direct_eval = []
    for idx in target.by_group.index:
        mask_A = g_A == idx[0]
        mask_B = g_B == idx[1]
        mask = np.logical_and(mask_A, mask_B)
        if idx == ('bb', 'x'):
            assert sum(mask) == 0, 'idx={0}'.format(idx)
        else:
            assert sum(mask) != 0, 'idx={0}'.format(idx)
            nxt = metric_fn(y_t[mask], y_p[mask])
            direct_eval.append(nxt)
    assert len(direct_eval) == 5

    # Check we have expected values
    assert target.by_group[metric_fn.__name__][('bb', 'x')] is None
    assert target.group_min()[metric_fn.__name__] == min(direct_eval)
    assert target.group_max()[metric_fn.__name__] == max(direct_eval)
    assert target.difference()[metric_fn.__name__] == max(direct_eval)-min(direct_eval)
    assert target.difference(method='to_overall')[metric_fn.__name__] == \
        max([abs(x-overall) for x in direct_eval])
    assert target.ratio()[metric_fn.__name__] == min(direct_eval) / max(direct_eval)
    assert target.ratio(method='to_overall')[metric_fn.__name__] == \
        min([x/overall for x in direct_eval] + [overall/x for x in direct_eval])
