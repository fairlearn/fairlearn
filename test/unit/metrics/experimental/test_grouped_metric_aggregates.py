# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics

from .data_for_test import y_t, y_p, g_1, g_2, g_3, g_4


def test_1m_1sf_0cf():
    target = metrics.GroupedMetric(skm.recall_score,
                                   y_t, y_p,
                                   sensitive_features=g_4)

    recall = skm.recall_score(y_t, y_p)
    mask_p = (g_4 == 'pp')
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

    target_diff = target.difference()
    assert isinstance(target_diff, pd.DataFrame)
    assert target_diff.shape == (1, 1)
    assert target_diff['recall_score']['overall'] == abs(recall_p - recall_q)

    target_diff_overall = target.difference(method='to_overall')
    assert isinstance(target_diff_overall, pd.DataFrame)
    assert target_diff_overall.shape == (1, 1)
    diffs_overall = [abs(recall_p-recall), abs(recall_q-recall)]
    assert target_diff_overall['recall_score']['overall'] == max(diffs_overall)


def test_2m_1sf_0cf():
    target = metrics.GroupedMetric({'recall': skm.recall_score, 'prec': skm.precision_score},
                                   y_t, y_p,
                                   sensitive_features=g_4)

    recall = skm.recall_score(y_t, y_p)
    prec = skm.precision_score(y_t, y_p)
    mask_p = (g_4 == 'pp')
    mask_q = (g_4 == 'q')
    recall_p = skm.recall_score(y_t[mask_p], y_p[mask_p])
    recall_q = skm.recall_score(y_t[mask_q], y_p[mask_q])
    prec_p = skm.precision_score(y_t[mask_p], y_p[mask_p])
    prec_q = skm.precision_score(y_t[mask_q], y_p[mask_q])

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_mins.shape == (1, 2)
    assert target_mins['recall']['overall'] == min(recall_p, recall_q)
    assert target_mins['prec']['overall'] == min(prec_p, prec_q)

    target_maxes = target.group_max()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_maxes.shape == (1, 2)
    assert target_maxes['recall']['overall'] == max(recall_p, recall_q)
    assert target_maxes['prec']['overall'] == max(prec_p, prec_q)

    target_diffs = target.difference()
    assert isinstance(target_diffs, pd.DataFrame)
    assert target_diffs.shape == (1, 2)
    assert target_diffs['recall']['overall'] == abs(recall_p - recall_q)
    assert target_diffs['prec']['overall'] == abs(prec_p - prec_q)

    target_diff_overall = target.difference(method='to_overall')
    assert isinstance(target_diff_overall, pd.DataFrame)
    assert target_diff_overall.shape == (1, 2)
    recall_diffs_overall = [abs(recall_p-recall), abs(recall_q-recall)]
    prec_diffs_overall = [abs(prec_p-prec), abs(prec_q-prec)]
    assert target_diff_overall['recall']['overall'] == max(recall_diffs_overall)
    assert target_diff_overall['prec']['overall'] == max(prec_diffs_overall)


def test_1m_1sf_1cf():
    target = metrics.GroupedMetric(skm.recall_score,
                                   y_t, y_p,
                                   sensitive_features=g_2,
                                   conditional_features=g_3)
    mask_f = (g_2 == 'f')
    mask_g = (g_2 == 'g')
    mask_k = (g_3 == 'kk')
    mask_m = (g_3 == 'm')

    mask_k_f = np.logical_and(mask_k, mask_f)
    mask_k_g = np.logical_and(mask_k, mask_g)
    mask_m_f = np.logical_and(mask_m, mask_f)
    mask_m_g = np.logical_and(mask_m, mask_g)
    recall_k = skm.recall_score(y_t[mask_k], y_p[mask_k])
    recall_m = skm.recall_score(y_t[mask_m], y_p[mask_m])
    recall_k_f = skm.recall_score(y_t[mask_k_f], y_p[mask_k_f])
    recall_m_f = skm.recall_score(y_t[mask_m_f], y_p[mask_m_f])
    recall_k_g = skm.recall_score(y_t[mask_k_g], y_p[mask_k_g])
    recall_m_g = skm.recall_score(y_t[mask_m_g], y_p[mask_m_g])
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

    target_diff = target.difference()
    assert isinstance(target_diff, pd.DataFrame)
    assert target_diff.shape == (2, 1)
    assert target_diff['recall_score']['kk'] == max(recall_k_arr)-min(recall_k_arr)
    assert target_diff['recall_score']['m'] == max(recall_m_arr)-min(recall_m_arr)

    target_diff_overall = target.difference(method='to_overall')
    assert isinstance(target_diff_overall, pd.DataFrame)
    assert target_diff_overall.shape == (2, 1)
    k_diffs_overall = [abs(recall_k_f-recall_k), abs(recall_k_g-recall_k)]
    m_diffs_overall = [abs(recall_m_f-recall_m), abs(recall_m_g-recall_m)]
    assert target_diff_overall['recall_score']['kk'] == max(k_diffs_overall)
    assert target_diff_overall['recall_score']['m'] == max(m_diffs_overall)


def test_1m_1sf_2cf():
    target = metrics.GroupedMetric(skm.recall_score,
                                   y_t, y_p,
                                   sensitive_features=g_2,
                                   conditional_features=[g_3, g_1])

    # Check we have correct return types
    assert isinstance(target.overall, pd.DataFrame)
    assert isinstance(target.by_group, pd.DataFrame)

    mask_a = (g_1 == 'aa')
    mask_b = (g_1 == 'ba')
    mask_f = (g_2 == 'f')
    mask_g = (g_2 == 'g')
    mask_k = (g_3 == 'kk')
    mask_m = (g_3 == 'm')

    mask_k_a = np.logical_and(mask_k, mask_a)
    mask_k_b = np.logical_and(mask_k, mask_b)
    mask_m_a = np.logical_and(mask_m, mask_a)
    mask_m_b = np.logical_and(mask_m, mask_b)
    mask_k_a_f = np.logical_and(mask_k_a, mask_f)
    mask_k_a_g = np.logical_and(mask_k_a, mask_g)
    mask_k_b_f = np.logical_and(mask_k_b, mask_f)
    mask_k_b_g = np.logical_and(mask_k_b, mask_g)
    mask_m_a_f = np.logical_and(mask_m_a, mask_f)
    mask_m_a_g = np.logical_and(mask_m_a, mask_g)
    mask_m_b_f = np.logical_and(mask_m_b, mask_f)
    mask_m_b_g = np.logical_and(mask_m_b, mask_g)

    recall_k_a = skm.recall_score(y_t[mask_k_a], y_p[mask_k_a])
    recall_k_b = skm.recall_score(y_t[mask_k_b], y_p[mask_k_b])
    recall_m_a = skm.recall_score(y_t[mask_m_a], y_p[mask_m_a])
    recall_m_b = skm.recall_score(y_t[mask_m_b], y_p[mask_m_b])
    recall_k_a_f = skm.recall_score(y_t[mask_k_a_f], y_p[mask_k_a_f])
    recall_k_a_g = skm.recall_score(y_t[mask_k_a_g], y_p[mask_k_a_g])
    recall_k_b_f = skm.recall_score(y_t[mask_k_b_f], y_p[mask_k_b_f])
    recall_k_b_g = skm.recall_score(y_t[mask_k_b_g], y_p[mask_k_b_g])
    recall_m_a_f = skm.recall_score(y_t[mask_m_a_f], y_p[mask_m_a_f])
    recall_m_a_g = skm.recall_score(y_t[mask_m_a_g], y_p[mask_m_a_g])
    recall_m_b_f = skm.recall_score(y_t[mask_m_b_f], y_p[mask_m_b_f])
    recall_m_b_g = skm.recall_score(y_t[mask_m_b_g], y_p[mask_m_b_g])

    recall_k_a_arr = [recall_k_a_f, recall_k_a_g]
    recall_k_b_arr = [recall_k_b_f, recall_k_b_g]
    recall_m_a_arr = [recall_m_a_f, recall_m_a_g]
    recall_m_b_arr = [recall_m_b_f, recall_m_b_g]

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_mins.shape == (4, 1)
    assert target_mins['recall_score'][('kk', 'aa')] == min(recall_k_a_arr)
    assert target_mins['recall_score'][('kk', 'ba')] == min(recall_k_b_arr)
    assert target_mins['recall_score'][('m', 'aa')] == min(recall_m_a_arr)
    assert target_mins['recall_score'][('m', 'ba')] == min(recall_m_b_arr)

    target_maxs = target.group_max()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_maxs.shape == (4, 1)
    assert target_maxs['recall_score'][('kk', 'aa')] == max(recall_k_a_arr)
    assert target_maxs['recall_score'][('kk', 'ba')] == max(recall_k_b_arr)
    assert target_maxs['recall_score'][('m', 'aa')] == max(recall_m_a_arr)
    assert target_maxs['recall_score'][('m', 'ba')] == max(recall_m_b_arr)

    diffs = target.difference()
    assert isinstance(diffs, pd.DataFrame)
    assert diffs.shape == (4, 1)
    assert diffs['recall_score'][('kk', 'aa')] == max(recall_k_a_arr) - min(recall_k_a_arr)
    assert diffs['recall_score'][('kk', 'ba')] == max(recall_k_b_arr) - min(recall_k_b_arr)
    assert diffs['recall_score'][('m', 'aa')] == max(recall_m_a_arr) - min(recall_m_a_arr)
    assert diffs['recall_score'][('m', 'ba')] == max(recall_m_b_arr) - min(recall_m_b_arr)

    diffs_overall = target.difference(method='to_overall')
    assert isinstance(diffs_overall, pd.DataFrame)
    assert diffs_overall.shape == (4, 1)
    diff_overall_k_a = max([abs(x-recall_k_a) for x in recall_k_a_arr])
    diff_overall_k_b = max([abs(x-recall_k_b) for x in recall_k_b_arr])
    diff_overall_m_a = max([abs(x-recall_m_a) for x in recall_m_a_arr])
    diff_overall_m_b = max([abs(x-recall_m_b) for x in recall_m_b_arr])

    assert diffs_overall['recall_score'][('kk', 'aa')] == diff_overall_k_a
    assert diffs_overall['recall_score'][('kk', 'ba')] == diff_overall_k_b
    assert diffs_overall['recall_score'][('m', 'aa')] == diff_overall_m_a
    assert diffs_overall['recall_score'][('m', 'ba')] == diff_overall_m_b


def test_2m_1sf_1cf():
    target = metrics.GroupedMetric({'recall': skm.recall_score, 'prec': skm.precision_score},
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

    mask_k_f = np.logical_and(mask_k, mask_f)
    mask_k_g = np.logical_and(mask_k, mask_g)
    mask_m_f = np.logical_and(mask_m, mask_f)
    mask_m_g = np.logical_and(mask_m, mask_g)
    recall_k = skm.recall_score(y_t[mask_k], y_p[mask_k])
    recall_m = skm.recall_score(y_t[mask_m], y_p[mask_m])
    recall_k_f = skm.recall_score(y_t[mask_k_f], y_p[mask_k_f])
    recall_m_f = skm.recall_score(y_t[mask_m_f], y_p[mask_m_f])
    recall_k_g = skm.recall_score(y_t[mask_k_g], y_p[mask_k_g])
    recall_m_g = skm.recall_score(y_t[mask_m_g], y_p[mask_m_g])
    recall_k_arr = [recall_k_f, recall_k_g]
    recall_m_arr = [recall_m_f, recall_m_g]
    precision_k = skm.precision_score(y_t[mask_k], y_p[mask_k])
    precision_m = skm.precision_score(y_t[mask_m], y_p[mask_m])
    precision_k_f = skm.precision_score(y_t[mask_k_f], y_p[mask_k_f])
    precision_m_f = skm.precision_score(y_t[mask_m_f], y_p[mask_m_f])
    precision_k_g = skm.precision_score(y_t[mask_k_g], y_p[mask_k_g])
    precision_m_g = skm.precision_score(y_t[mask_m_g], y_p[mask_m_g])
    precision_k_arr = [precision_k_f, precision_k_g]
    precision_m_arr = [precision_m_f, precision_m_g]

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_mins.shape == (2, 2)
    assert target_mins['recall']['kk'] == min(recall_k_arr)
    assert target_mins['recall']['m'] == min(recall_m_arr)
    assert target_mins['prec']['kk'] == min(precision_k_arr)
    assert target_mins['prec']['m'] == min(precision_m_arr)

    target_maxs = target.group_max()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_maxs.shape == (2, 2)
    assert target_maxs['recall']['kk'] == max(recall_k_arr)
    assert target_maxs['recall']['m'] == max(recall_m_arr)
    assert target_maxs['prec']['kk'] == max(precision_k_arr)
    assert target_maxs['prec']['m'] == max(precision_m_arr)

    diffs = target.difference()
    assert isinstance(diffs, pd.DataFrame)
    assert diffs.shape == (2, 2)
    assert diffs['recall']['kk'] == max(recall_k_arr) - min(recall_k_arr)
    assert diffs['recall']['m'] == max(recall_m_arr) - min(recall_m_arr)
    assert diffs['prec']['kk'] == max(precision_k_arr) - min(precision_k_arr)
    assert diffs['prec']['m'] == max(precision_m_arr) - min(precision_m_arr)

    diffs_overall = target.difference(method='to_overall')
    assert isinstance(diffs_overall, pd.DataFrame)
    assert diffs_overall.shape == (2, 2)
    recall_k_overall = max([abs(x-recall_k) for x in recall_k_arr])
    recall_m_overall = max([abs(x-recall_m) for x in recall_m_arr])
    precision_k_overall = max([abs(x-precision_k) for x in precision_k_arr])
    precision_m_overall = max([abs(x-precision_m) for x in precision_m_arr])
    assert diffs_overall['recall']['kk'] == recall_k_overall
    assert diffs_overall['recall']['m'] == recall_m_overall
    assert diffs_overall['prec']['kk'] == precision_k_overall
    assert diffs_overall['prec']['m'] == precision_m_overall
