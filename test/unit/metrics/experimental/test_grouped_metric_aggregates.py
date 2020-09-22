# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics

from .data_for_test import y_t, y_p, s_w, g_1, g_2, g_3, g_4


def test_1m_1sf_0cf():
    target = metrics.GroupedMetric(skm.recall_score,
                                   y_t, y_p,
                                   sensitive_features=g_4)

    mask_p = (g_4 == 'p')
    mask_q = (g_4 == 'q')
    recall_p = skm.recall_score(y_t[mask_p], y_p[mask_p])
    recall_q = skm.recall_score(y_t[mask_q], y_p[mask_q])

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_mins.shape == (1, 1)
    assert target_mins['recall_score']['overall'] == min(recall_p, recall_q)

    target_maxes = target.group_max()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_maxes.shape == (1, 1)
    assert target_maxes['recall_score']['overall'] == max(recall_p, recall_q)


def test_2m_1sf_0cf():
    target = metrics.GroupedMetric([skm.recall_score, skm.precision_score],
                                   y_t, y_p,
                                   sensitive_features=g_4)

    mask_p = (g_4 == 'p')
    mask_q = (g_4 == 'q')
    recall_p = skm.recall_score(y_t[mask_p], y_p[mask_p])
    recall_q = skm.recall_score(y_t[mask_q], y_p[mask_q])
    prec_p = skm.precision_score(y_t[mask_p], y_p[mask_p])
    prec_q = skm.precision_score(y_t[mask_q], y_p[mask_q])

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_mins.shape == (1, 2)
    assert target_mins['recall_score']['overall'] == min(recall_p, recall_q)
    assert target_mins['precision_score']['overall'] == min(prec_p, prec_q)

    target_maxes = target.group_max()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_maxes.shape == (1, 2)
    assert target_maxes['recall_score']['overall'] == max(recall_p, recall_q)
    assert target_maxes['precision_score']['overall'] == max(prec_p, prec_q)


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
    mask_k = (g_3 == 'k')
    mask_m = (g_3 == 'm')

    mask_k_f = np.logical_and(mask_k, mask_f)
    mask_k_g = np.logical_and(mask_k, mask_g)
    mask_m_f = np.logical_and(mask_m, mask_f)
    mask_m_g = np.logical_and(mask_m, mask_g)
    recall_k_f = skm.recall_score(y_t[mask_k_f], y_p[mask_k_f])
    recall_m_f = skm.recall_score(y_t[mask_m_f], y_p[mask_m_f])
    recall_k_g = skm.recall_score(y_t[mask_k_g], y_p[mask_k_g])
    recall_m_g = skm.recall_score(y_t[mask_m_g], y_p[mask_m_g])
    recall_k_arr = [recall_k_f, recall_k_g]
    recall_m_arr = [recall_m_f, recall_m_g]

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_mins.shape == (2, 1)
    assert target_mins['recall_score']['k'] == min(recall_k_arr)
    assert target_mins['recall_score']['m'] == min(recall_m_arr)

    target_maxs = target.group_max()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_maxs.shape == (2, 1)
    assert target_maxs['recall_score']['k'] == max(recall_k_arr)
    assert target_maxs['recall_score']['m'] == max(recall_m_arr)


def test_2m_1sf_1cf():
    target = metrics.GroupedMetric([skm.recall_score, skm.precision_score],
                                   y_t, y_p,
                                   sensitive_features=g_2,
                                   conditional_features=g_3)

    # Check we have correct return types
    assert isinstance(target.overall, pd.DataFrame)
    assert isinstance(target.by_group, pd.DataFrame)

    mask_f = (g_2 == 'f')
    mask_g = (g_2 == 'g')
    mask_k = (g_3 == 'k')
    mask_m = (g_3 == 'm')

    mask_k_f = np.logical_and(mask_k, mask_f)
    mask_k_g = np.logical_and(mask_k, mask_g)
    mask_m_f = np.logical_and(mask_m, mask_f)
    mask_m_g = np.logical_and(mask_m, mask_g)
    recall_k_f = skm.recall_score(y_t[mask_k_f], y_p[mask_k_f])
    recall_m_f = skm.recall_score(y_t[mask_m_f], y_p[mask_m_f])
    recall_k_g = skm.recall_score(y_t[mask_k_g], y_p[mask_k_g])
    recall_m_g = skm.recall_score(y_t[mask_m_g], y_p[mask_m_g])
    recall_k_arr = [recall_k_f, recall_k_g]
    recall_m_arr = [recall_m_f, recall_m_g]
    precision_k_f = skm.precision_score(y_t[mask_k_f], y_p[mask_k_f])
    precision_m_f = skm.precision_score(y_t[mask_m_f], y_p[mask_m_f])
    precision_k_g = skm.precision_score(y_t[mask_k_g], y_p[mask_k_g])
    precision_m_g = skm.precision_score(y_t[mask_m_g], y_p[mask_m_g])
    precision_k_arr = [precision_k_f, precision_k_g]
    precision_m_arr = [precision_m_f, precision_m_g]

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_mins.shape == (2, 2)
    assert target_mins['recall_score']['k'] == min(recall_k_arr)
    assert target_mins['recall_score']['m'] == min(recall_m_arr)
    assert target_mins['precision_score']['k'] == min(precision_k_arr)
    assert target_mins['precision_score']['m'] == min(precision_m_arr)

    target_maxs = target.group_max()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_maxs.shape == (2, 2)
    assert target_maxs['recall_score']['k'] == max(recall_k_arr)
    assert target_maxs['recall_score']['m'] == max(recall_m_arr)
    assert target_maxs['precision_score']['k'] == max(precision_k_arr)
    assert target_maxs['precision_score']['m'] == max(precision_m_arr)
