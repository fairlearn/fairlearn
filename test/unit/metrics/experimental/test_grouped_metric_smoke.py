# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics

from .data_for_test import y_t, y_p, s_w, g_1, g_2, g_3, g_4


def test_basic():
    g_f = pd.DataFrame(data=g_4, columns=['My feature'])
    target = metrics.GroupedMetric(skm.recall_score,
                                   y_t, y_p,
                                   sensitive_features=g_f)

    # Check we have correct return types
    assert isinstance(target.overall, pd.DataFrame)
    assert isinstance(target.by_group, pd.DataFrame)

    # Check we have expected number of elements
    assert target.overall.shape == (1, 1)
    assert target.by_group.shape == (2, 1)
    assert np.array_equal(target.by_group.index.names, ['My feature'])

    recall_overall = skm.recall_score(y_t, y_p)
    assert target.overall['recall_score']['overall'] == recall_overall

    mask_p = (g_4 == 'p')
    mask_q = (g_4 == 'q')
    recall_p = skm.recall_score(y_t[mask_p], y_p[mask_p])
    recall_q = skm.recall_score(y_t[mask_q], y_p[mask_q])
    assert target.by_group['recall_score'][('p',)] == recall_p
    assert target.by_group['recall_score'][('q',)] == recall_q


'''
    max_vals = target.group_max()
    assert isinstance(max_vals, pd.Series)
    assert len(max_vals) == 1
    assert max_vals[('recall_score',)] == max([expected_A, expected_B])

    min_vals = target.group_min()
    assert isinstance(min_vals, pd.Series)
    assert len(min_vals) == 1
    assert min_vals[('recall_score',)] == min([expected_A, expected_B])

    diffs = target.difference()
    assert isinstance(diffs, pd.Series)
    assert len(diffs) == 1
    assert diffs[('recall_score',)] == abs(expected_A - expected_B)

    diff_overall = target.difference_to_overall()
    assert isinstance(diff_overall, pd.Series)
    assert len(diff_overall) == 1
    expected_diff_overall = max([abs(expected_A-expected_overall),
                                 abs(expected_B-expected_overall)])
    assert diff_overall[('recall_score',)] == expected_diff_overall
'''
