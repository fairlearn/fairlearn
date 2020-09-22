# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics

from .data_for_test import y_t, y_p, g_2, g_3, g_4


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

    mask_p = (g_4 == 'pp')
    mask_q = (g_4 == 'q')
    recall_p = skm.recall_score(y_t[mask_p], y_p[mask_p])
    recall_q = skm.recall_score(y_t[mask_q], y_p[mask_q])
    assert target.by_group['recall_score']['pp'] == recall_p
    assert target.by_group['recall_score']['q'] == recall_q

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_mins.shape == (1, 1)
    assert target_mins['recall_score']['overall'] == min(recall_p, recall_q)

    target_maxes = target.group_max()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_maxes.shape == (1, 1)
    assert target_maxes['recall_score']['overall'] == max(recall_p, recall_q)


def test_1m_1sf_1cf():
    target = metrics.GroupedMetric(skm.recall_score,
                                   y_t, y_p,
                                   sensitive_features=g_2,
                                   conditional_features=g_3)

    # Check we have correct return types
    assert isinstance(target.overall, pd.DataFrame)
    assert isinstance(target.by_group, pd.DataFrame)

    mask_f = (g_2 == 'f')
    mask_g = (g_2 == 'g')
    mask_k = (g_3 == 'kk')
    mask_m = (g_3 == 'm')

    # Check we have expected number of elements
    assert target.overall.shape == (2, 1)
    assert target.by_group.shape == (4, 1)

    recall_k = skm.recall_score(y_t[mask_k], y_p[mask_k])
    recall_m = skm.recall_score(y_t[mask_m], y_p[mask_m])
    assert target.overall['recall_score']['kk'] == recall_k
    assert target.overall['recall_score']['m'] == recall_m

    mask_k_f = np.logical_and(mask_k, mask_f)
    mask_k_g = np.logical_and(mask_k, mask_g)
    mask_m_f = np.logical_and(mask_m, mask_f)
    mask_m_g = np.logical_and(mask_m, mask_g)
    recall_k_f = skm.recall_score(y_t[mask_k_f], y_p[mask_k_f])
    recall_m_f = skm.recall_score(y_t[mask_m_f], y_p[mask_m_f])
    recall_k_g = skm.recall_score(y_t[mask_k_g], y_p[mask_k_g])
    recall_m_g = skm.recall_score(y_t[mask_m_g], y_p[mask_m_g])
    assert target.by_group['recall_score'][('kk', 'f')] == recall_k_f
    assert target.by_group['recall_score'][('kk', 'g')] == recall_k_g
    assert target.by_group['recall_score'][('m', 'f')] == recall_m_f
    assert target.by_group['recall_score'][('m', 'g')] == recall_m_g

    recall_k_arr = [recall_k_f, recall_k_g]
    recall_m_arr = [recall_m_f, recall_m_g]

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_mins.shape == (2, 1)
    assert target_mins['recall_score']['kk'] == min(recall_k_arr)
    assert target_mins['recall_score']['m'] == min(recall_m_arr)

    target_maxs = target.group_max()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_maxs.shape == (2, 1)
    assert target_maxs['recall_score']['kk'] == max(recall_k_arr)
    assert target_maxs['recall_score']['m'] == max(recall_m_arr)
