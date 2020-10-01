# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics

from .data_for_test import y_t, y_p, group_gen, g_1

metric_functions = [skm.recall_score,
                    skm.precision_score,
                    skm.accuracy_score,
                    skm.balanced_accuracy_score]


n = len(y_t)
g_A = np.asarray([group_gen(x, int(n/2), ['aa', 'bb']) for x in range(n)])
g_B = np.asarray([group_gen(x, 1+int(n/3), ['x', 'y', 'z']) for x in range(n)])


@pytest.mark.parametrize("metric_fn", metric_functions)
def test_missing_sensitive_feature_combinations(metric_fn):

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


def test_missing_conditional_feature_combinations():
    metric_fn = skm.accuracy_score

    target = metrics.GroupedMetric(metric_fn,
                                   y_t, y_p,
                                   conditional_features=[g_A, g_B],
                                   sensitive_features=g_1)

    # Build all the expected values
    overall = dict()
    by_groups = dict()
    for i_A in np.unique(g_A):
        mask_A = g_A == i_A

        overall_B = dict()
        by_groups_B = dict()
        for i_B in np.unique(g_B):
            mask_B = g_B == i_B

            mask_A_B = np.logical_and(mask_A, mask_B)
            if i_A == 'bb' and i_B == 'x':
                assert sum(mask_A_B) == 0
            else:
                assert sum(mask_A_B) != 0
                overall_B[i_B] = metric_fn(y_t[mask_A_B], y_p[mask_A_B])
                b_g = dict()
                for i_1 in np.unique(g_1):
                    mask = np.logical_and(mask_A_B, g_1 == i_1)
                    if sum(mask) != 0:
                        b_g[i_1] = metric_fn(y_t[mask], y_p[mask])
                by_groups_B[i_B] = b_g
        overall[i_A] = overall_B
        by_groups[i_A] = by_groups_B

    # Check values are as expected
    for i_A in np.unique(g_A):
        for i_B in np.unique(g_B):
            if i_B in overall[i_A]:
                assert overall[i_A][i_B] == \
                    target.overall[metric_fn.__name__][(i_A, i_B)]
                for i_1 in np.unique(g_1):
                    if i_1 in by_groups[i_A][i_B]:
                        assert by_groups[i_A][i_B][i_1] == \
                            target.by_group[metric_fn.__name__][(i_A, i_B, i_1)]
                    else:
                        assert target.by_group[metric_fn.__name__][(i_A, i_B, i_1)] is None
            else:
                assert target.overall[metric_fn.__name__][(i_A, i_B)] is None

    # Check differences and ratios
    for i_A in np.unique(g_A):
        for i_B in np.unique(g_B):
            if i_B in overall[i_A]:
                expected_max = max(by_groups[i_A][i_B].values())
                expected_min = min(by_groups[i_A][i_B].values())
                assert expected_max == target.group_max()[metric_fn.__name__][(i_A, i_B)]
                assert expected_min == target.group_min()[metric_fn.__name__][(i_A, i_B)]
            else:
                assert target.group_max()[metric_fn.__name__][(i_A, i_B)] is None
                assert target.group_min()[metric_fn.__name__][(i_A, i_B)] is None
