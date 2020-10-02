# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics

from .data_for_test import y_t, y_p, g_1, g_2, g_3, g_4


metric_functions = [skm.recall_score,
                    skm.precision_score,
                    skm.accuracy_score,
                    skm.balanced_accuracy_score]


@pytest.mark.parametrize("metric_fn", metric_functions)
def test_1m_1sf_0cf(metric_fn):
    target = metrics.GroupedMetric(metric_fn,
                                   y_t, y_p,
                                   sensitive_features=g_4)

    overall = metric_fn(y_t, y_p)
    mask_p = (g_4 == 'pp')
    mask_q = (g_4 == 'q')
    metric_p = metric_fn(y_t[mask_p], y_p[mask_p])
    metric_q = metric_fn(y_t[mask_q], y_p[mask_q])

    mfn = metric_fn.__name__

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.Series)
    assert len(target_mins) == 1
    assert target_mins[mfn] == min(metric_p, metric_q)

    target_maxes = target.group_max()
    assert isinstance(target_mins, pd.Series)
    assert len(target_maxes) == 1
    assert target_maxes[mfn] == max(metric_p, metric_q)

    target_diff = target.difference()
    assert isinstance(target_diff, pd.Series)
    assert len(target_diff) == 1
    assert target_diff[mfn] == abs(metric_p - metric_q)

    target_diff_overall = target.difference(method='to_overall')
    assert isinstance(target_diff_overall, pd.Series)
    assert len(target_diff_overall) == 1
    diffs_overall = [abs(metric_p-overall), abs(metric_q-overall)]
    assert target_diff_overall[mfn] == max(diffs_overall)

    target_ratio = target.ratio()
    assert isinstance(target_ratio, pd.Series)
    assert len(target_ratio) == 1
    assert target_ratio[mfn] == min(metric_p, metric_q)/max(metric_p, metric_q)

    target_ratio_overall = target.ratio(method='to_overall')
    assert isinstance(target_ratio_overall, pd.Series)
    assert len(target_ratio_overall) == 1
    expected_ratio_overall = min(metric_p/overall,
                                 overall/metric_p,
                                 metric_q/overall,
                                 overall/metric_q)
    # Need to use approx, due to internal method of calculating
    # the correct ratio. Internally, GroupedMetric computes a ratio
    # and takes the reciprocal if it's greater than 1
    assert target_ratio_overall[mfn] == pytest.approx(expected_ratio_overall,
                                                      rel=1e-10, abs=1e-16)


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
    assert isinstance(target_mins, pd.Series)
    assert len(target_mins) == 2
    assert target_mins['recall'] == min(recall_p, recall_q)
    assert target_mins['prec'] == min(prec_p, prec_q)

    target_maxes = target.group_max()
    assert isinstance(target_mins, pd.Series)
    assert isinstance(target_maxes, pd.Series)
    assert target_maxes['recall'] == max(recall_p, recall_q)
    assert target_maxes['prec'] == max(prec_p, prec_q)

    target_diffs = target.difference()
    assert isinstance(target_diffs, pd.Series)
    assert len(target_diffs) == 2
    assert target_diffs['recall'] == abs(recall_p - recall_q)
    assert target_diffs['prec'] == abs(prec_p - prec_q)

    target_diff_overall = target.difference(method='to_overall')
    assert isinstance(target_diff_overall, pd.Series)
    assert len(target_diff_overall) == 2
    recall_diffs_overall = [abs(recall_p-recall), abs(recall_q-recall)]
    prec_diffs_overall = [abs(prec_p-prec), abs(prec_q-prec)]
    assert target_diff_overall['recall'] == max(recall_diffs_overall)
    assert target_diff_overall['prec'] == max(prec_diffs_overall)

    target_ratio = target.ratio()
    assert isinstance(target_ratio, pd.Series)
    assert len(target_ratio) == 2
    assert target_ratio['recall'] == min(recall_p, recall_q)/max(recall_p, recall_q)
    assert target_ratio['prec'] == min(prec_p/prec_q, prec_q/prec_p)

    target_ratio_overall = target.ratio(method='to_overall')
    assert isinstance(target_ratio_overall, pd.Series)
    assert len(target_ratio_overall) == 2
    recall_ratio_overall = min(recall_p/recall, recall/recall_p,
                               recall_q/recall, recall/recall_q)
    prec_ratio_overall = min(prec_p/prec, prec/prec_p,
                             prec_q/prec, prec/prec_q)
    # Need pytest.approx because ratio() doesn't do quite the same
    # calculations as above. Specifically, rather than doing both
    # preq_q/prec and prec/prec_q, it only calculates the former,
    # and inverts it if the result is greater than unity
    assert target_ratio_overall['recall'] == pytest.approx(recall_ratio_overall,
                                                           rel=1e-10, abs=1e-16)
    assert target_ratio_overall['prec'] == pytest.approx(prec_ratio_overall,
                                                         rel=1e-10, abs=1e-16)


@pytest.mark.parametrize("metric_fn", metric_functions)
def test_1m_1sf_1cf(metric_fn):
    target = metrics.GroupedMetric(metric_fn,
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
    metric_k = metric_fn(y_t[mask_k], y_p[mask_k])
    metric_m = metric_fn(y_t[mask_m], y_p[mask_m])
    metric_k_f = metric_fn(y_t[mask_k_f], y_p[mask_k_f])
    metric_m_f = metric_fn(y_t[mask_m_f], y_p[mask_m_f])
    metric_k_g = metric_fn(y_t[mask_k_g], y_p[mask_k_g])
    metric_m_g = metric_fn(y_t[mask_m_g], y_p[mask_m_g])
    metric_k_arr = [metric_k_f, metric_k_g]
    metric_m_arr = [metric_m_f, metric_m_g]

    mfn = metric_fn.__name__

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_mins.shape == (2, 1)
    assert target_mins[mfn]['kk'] == min(metric_k_arr)
    assert target_mins[mfn]['m'] == min(metric_m_arr)

    target_maxs = target.group_max()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_maxs.shape == (2, 1)
    assert target_maxs[mfn]['kk'] == max(metric_k_arr)
    assert target_maxs[mfn]['m'] == max(metric_m_arr)

    target_diff = target.difference()
    assert isinstance(target_diff, pd.DataFrame)
    assert target_diff.shape == (2, 1)
    assert target_diff[mfn]['kk'] == max(metric_k_arr)-min(metric_k_arr)
    assert target_diff[mfn]['m'] == max(metric_m_arr)-min(metric_m_arr)

    target_diff_overall = target.difference(method='to_overall')
    assert isinstance(target_diff_overall, pd.DataFrame)
    assert target_diff_overall.shape == (2, 1)
    k_diffs_overall = [abs(metric_k_f-metric_k), abs(metric_k_g-metric_k)]
    m_diffs_overall = [abs(metric_m_f-metric_m), abs(metric_m_g-metric_m)]
    assert target_diff_overall[mfn]['kk'] == max(k_diffs_overall)
    assert target_diff_overall[mfn]['m'] == max(m_diffs_overall)

    target_ratio = target.ratio()
    assert isinstance(target_ratio, pd.DataFrame)
    assert target_ratio.shape == (2, 1)
    assert target_ratio[mfn]['kk'] == min(metric_k_arr)/max(metric_k_arr)
    assert target_ratio[mfn]['m'] == min(metric_m_arr)/max(metric_m_arr)

    target_ratio_overall = target.ratio(method='to_overall')
    assert isinstance(target_ratio_overall, pd.DataFrame)
    assert target_ratio_overall.shape == (2, 1)
    k_ratios_overall = [x/metric_k for x in metric_k_arr] + \
        [metric_k/x for x in metric_k_arr]
    m_ratios_overall = [x/metric_m for x in metric_m_arr] + \
        [metric_m/x for x in metric_m_arr]
    # Ratio to overall is forced to be <1 in a slightly different way
    # internally, so have to use pytest.approx
    assert target_ratio_overall[mfn]['kk'] == pytest.approx(min(k_ratios_overall),
                                                            rel=1e-10, abs=1e-16)
    assert target_ratio_overall[mfn]['m'] == pytest.approx(min(m_ratios_overall),
                                                           rel=1e-10, abs=1e-16)


@pytest.mark.parametrize("metric_fn", metric_functions)
def test_1m_1sf_2cf(metric_fn):
    target = metrics.GroupedMetric(metric_fn,
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

    metric_k_a = metric_fn(y_t[mask_k_a], y_p[mask_k_a])
    metric_k_b = metric_fn(y_t[mask_k_b], y_p[mask_k_b])
    metric_m_a = metric_fn(y_t[mask_m_a], y_p[mask_m_a])
    metric_m_b = metric_fn(y_t[mask_m_b], y_p[mask_m_b])
    metric_k_a_f = metric_fn(y_t[mask_k_a_f], y_p[mask_k_a_f])
    metric_k_a_g = metric_fn(y_t[mask_k_a_g], y_p[mask_k_a_g])
    metric_k_b_f = metric_fn(y_t[mask_k_b_f], y_p[mask_k_b_f])
    metric_k_b_g = metric_fn(y_t[mask_k_b_g], y_p[mask_k_b_g])
    metric_m_a_f = metric_fn(y_t[mask_m_a_f], y_p[mask_m_a_f])
    metric_m_a_g = metric_fn(y_t[mask_m_a_g], y_p[mask_m_a_g])
    metric_m_b_f = metric_fn(y_t[mask_m_b_f], y_p[mask_m_b_f])
    metric_m_b_g = metric_fn(y_t[mask_m_b_g], y_p[mask_m_b_g])

    metric_k_a_arr = [metric_k_a_f, metric_k_a_g]
    metric_k_b_arr = [metric_k_b_f, metric_k_b_g]
    metric_m_a_arr = [metric_m_a_f, metric_m_a_g]
    metric_m_b_arr = [metric_m_b_f, metric_m_b_g]

    mfn = metric_fn.__name__

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_mins.shape == (4, 1)
    assert target_mins[mfn][('kk', 'aa')] == min(metric_k_a_arr)
    assert target_mins[mfn][('kk', 'ba')] == min(metric_k_b_arr)
    assert target_mins[mfn][('m', 'aa')] == min(metric_m_a_arr)
    assert target_mins[mfn][('m', 'ba')] == min(metric_m_b_arr)

    target_maxs = target.group_max()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_maxs.shape == (4, 1)
    assert target_maxs[mfn][('kk', 'aa')] == max(metric_k_a_arr)
    assert target_maxs[mfn][('kk', 'ba')] == max(metric_k_b_arr)
    assert target_maxs[mfn][('m', 'aa')] == max(metric_m_a_arr)
    assert target_maxs[mfn][('m', 'ba')] == max(metric_m_b_arr)

    diffs = target.difference()
    assert isinstance(diffs, pd.DataFrame)
    assert diffs.shape == (4, 1)
    assert diffs[mfn][('kk', 'aa')] == max(metric_k_a_arr) - min(metric_k_a_arr)
    assert diffs[mfn][('kk', 'ba')] == max(metric_k_b_arr) - min(metric_k_b_arr)
    assert diffs[mfn][('m', 'aa')] == max(metric_m_a_arr) - min(metric_m_a_arr)
    assert diffs[mfn][('m', 'ba')] == max(metric_m_b_arr) - min(metric_m_b_arr)

    diffs_overall = target.difference(method='to_overall')
    assert isinstance(diffs_overall, pd.DataFrame)
    assert diffs_overall.shape == (4, 1)
    diff_overall_k_a = max([abs(x-metric_k_a) for x in metric_k_a_arr])
    diff_overall_k_b = max([abs(x-metric_k_b) for x in metric_k_b_arr])
    diff_overall_m_a = max([abs(x-metric_m_a) for x in metric_m_a_arr])
    diff_overall_m_b = max([abs(x-metric_m_b) for x in metric_m_b_arr])

    assert diffs_overall[mfn][('kk', 'aa')] == diff_overall_k_a
    assert diffs_overall[mfn][('kk', 'ba')] == diff_overall_k_b
    assert diffs_overall[mfn][('m', 'aa')] == diff_overall_m_a
    assert diffs_overall[mfn][('m', 'ba')] == diff_overall_m_b

    ratios = target.ratio()
    assert isinstance(ratios, pd.DataFrame)
    assert ratios.shape == (4, 1)
    assert ratios[mfn][('kk', 'aa')] == min(metric_k_a_arr) / max(metric_k_a_arr)
    assert ratios[mfn][('kk', 'ba')] == min(metric_k_b_arr) / max(metric_k_b_arr)
    assert ratios[mfn][('m', 'aa')] == min(metric_m_a_arr) / max(metric_m_a_arr)
    assert ratios[mfn][('m', 'ba')] == min(metric_m_b_arr) / max(metric_m_b_arr)

    ratios_overall = target.ratio(method='to_overall')
    assert isinstance(ratios_overall, pd.DataFrame)
    assert ratios_overall.shape == (4, 1)
    ratio_overall_k_a = [x/metric_k_a for x in metric_k_a_arr] + \
        [metric_k_a/x for x in metric_k_a_arr]
    ratio_overall_k_b = [x/metric_k_b for x in metric_k_b_arr] + \
        [metric_k_b/x for x in metric_k_b_arr]
    ratio_overall_m_a = [x/metric_m_a for x in metric_m_a_arr] + \
        [metric_m_a/x for x in metric_m_a_arr]
    ratio_overall_m_b = [x/metric_m_b for x in metric_m_b_arr] + \
        [metric_m_b/x for x in metric_m_b_arr]
    assert ratios_overall[mfn][('kk', 'aa')] == min(ratio_overall_k_a)
    assert ratios_overall[mfn][('kk', 'ba')] == min(ratio_overall_k_b)
    assert ratios_overall[mfn][('m', 'aa')] == min(ratio_overall_m_a)
    assert ratios_overall[mfn][('m', 'ba')] == pytest.approx(min(ratio_overall_m_b),
                                                             rel=1e-10, abs=1e-16)


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

    ratios = target.ratio()
    assert isinstance(ratios, pd.DataFrame)
    assert ratios.shape == (2, 2)
    assert ratios['recall']['kk'] == min(recall_k_arr) / max(recall_k_arr)
    assert ratios['recall']['m'] == min(recall_m_arr) / max(recall_m_arr)
    assert ratios['prec']['kk'] == min(precision_k_arr) / max(precision_k_arr)
    assert ratios['prec']['m'] == min(precision_m_arr) / max(precision_m_arr)

    ratios_overall = target.ratio(method='to_overall')
    assert isinstance(ratios_overall, pd.DataFrame)
    assert ratios_overall.shape == (2, 2)
    recall_k_overall = [x/recall_k for x in recall_k_arr] + \
        [recall_k/x for x in recall_k_arr]
    recall_m_overall = [x/recall_m for x in recall_m_arr] + \
        [recall_m/x for x in recall_m_arr]
    precision_k_overall = [x/precision_k for x in precision_k_arr] + \
        [precision_k/x for x in precision_k_arr]
    precision_m_overall = [x/precision_m for x in precision_m_arr] + \
        [precision_m/x for x in precision_m_arr]
    assert ratios_overall['recall']['kk'] == min(recall_k_overall)
    assert ratios_overall['recall']['m'] == min(recall_m_overall)
    assert ratios_overall['prec']['kk'] == min(precision_k_overall)
    assert ratios_overall['prec']['m'] == pytest.approx(min(precision_m_overall),
                                                        rel=1e-10, abs=1e-16)


def test_2m_1sf_2cf():
    func_dict = {'recall': skm.recall_score, 'prec': skm.precision_score}
    target = metrics.GroupedMetric(func_dict,
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

    for mfn, metric_fn in func_dict.items():
        metric_k_a = metric_fn(y_t[mask_k_a], y_p[mask_k_a])
        metric_k_b = metric_fn(y_t[mask_k_b], y_p[mask_k_b])
        metric_m_a = metric_fn(y_t[mask_m_a], y_p[mask_m_a])
        metric_m_b = metric_fn(y_t[mask_m_b], y_p[mask_m_b])
        metric_k_a_f = metric_fn(y_t[mask_k_a_f], y_p[mask_k_a_f])
        metric_k_a_g = metric_fn(y_t[mask_k_a_g], y_p[mask_k_a_g])
        metric_k_b_f = metric_fn(y_t[mask_k_b_f], y_p[mask_k_b_f])
        metric_k_b_g = metric_fn(y_t[mask_k_b_g], y_p[mask_k_b_g])
        metric_m_a_f = metric_fn(y_t[mask_m_a_f], y_p[mask_m_a_f])
        metric_m_a_g = metric_fn(y_t[mask_m_a_g], y_p[mask_m_a_g])
        metric_m_b_f = metric_fn(y_t[mask_m_b_f], y_p[mask_m_b_f])
        metric_m_b_g = metric_fn(y_t[mask_m_b_g], y_p[mask_m_b_g])

        metric_k_a_arr = [metric_k_a_f, metric_k_a_g]
        metric_k_b_arr = [metric_k_b_f, metric_k_b_g]
        metric_m_a_arr = [metric_m_a_f, metric_m_a_g]
        metric_m_b_arr = [metric_m_b_f, metric_m_b_g]

        target_mins = target.group_min()
        assert isinstance(target_mins, pd.DataFrame)
        assert target_mins.shape == (4, 2)
        assert target_mins[mfn][('kk', 'aa')] == min(metric_k_a_arr)
        assert target_mins[mfn][('kk', 'ba')] == min(metric_k_b_arr)
        assert target_mins[mfn][('m', 'aa')] == min(metric_m_a_arr)
        assert target_mins[mfn][('m', 'ba')] == min(metric_m_b_arr)

        target_maxs = target.group_max()
        assert isinstance(target_mins, pd.DataFrame)
        assert target_maxs.shape == (4, 2)
        assert target_maxs[mfn][('kk', 'aa')] == max(metric_k_a_arr)
        assert target_maxs[mfn][('kk', 'ba')] == max(metric_k_b_arr)
        assert target_maxs[mfn][('m', 'aa')] == max(metric_m_a_arr)
        assert target_maxs[mfn][('m', 'ba')] == max(metric_m_b_arr)

        diffs = target.difference()
        assert isinstance(diffs, pd.DataFrame)
        assert diffs.shape == (4, 2)
        assert diffs[mfn][('kk', 'aa')] == max(metric_k_a_arr) - min(metric_k_a_arr)
        assert diffs[mfn][('kk', 'ba')] == max(metric_k_b_arr) - min(metric_k_b_arr)
        assert diffs[mfn][('m', 'aa')] == max(metric_m_a_arr) - min(metric_m_a_arr)
        assert diffs[mfn][('m', 'ba')] == max(metric_m_b_arr) - min(metric_m_b_arr)

        diffs_overall = target.difference(method='to_overall')
        assert isinstance(diffs_overall, pd.DataFrame)
        assert diffs_overall.shape == (4, 2)
        diff_overall_k_a = max([abs(x-metric_k_a) for x in metric_k_a_arr])
        diff_overall_k_b = max([abs(x-metric_k_b) for x in metric_k_b_arr])
        diff_overall_m_a = max([abs(x-metric_m_a) for x in metric_m_a_arr])
        diff_overall_m_b = max([abs(x-metric_m_b) for x in metric_m_b_arr])

        assert diffs_overall[mfn][('kk', 'aa')] == diff_overall_k_a
        assert diffs_overall[mfn][('kk', 'ba')] == diff_overall_k_b
        assert diffs_overall[mfn][('m', 'aa')] == diff_overall_m_a
        assert diffs_overall[mfn][('m', 'ba')] == diff_overall_m_b

        ratios = target.ratio()
        assert isinstance(ratios, pd.DataFrame)
        assert ratios.shape == (4, 2)
        assert ratios[mfn][('kk', 'aa')] == min(metric_k_a_arr) / max(metric_k_a_arr)
        assert ratios[mfn][('kk', 'ba')] == min(metric_k_b_arr) / max(metric_k_b_arr)
        assert ratios[mfn][('m', 'aa')] == min(metric_m_a_arr) / max(metric_m_a_arr)
        assert ratios[mfn][('m', 'ba')] == min(metric_m_b_arr) / max(metric_m_b_arr)

        ratios_overall = target.ratio(method='to_overall')
        assert isinstance(ratios_overall, pd.DataFrame)
        assert ratios_overall.shape == (4, 2)
        ratio_overall_k_a = [x/metric_k_a for x in metric_k_a_arr] + \
            [metric_k_a/x for x in metric_k_a_arr]
        ratio_overall_k_b = [x/metric_k_b for x in metric_k_b_arr] + \
            [metric_k_b/x for x in metric_k_b_arr]
        ratio_overall_m_a = [x/metric_m_a for x in metric_m_a_arr] + \
            [metric_m_a/x for x in metric_m_a_arr]
        ratio_overall_m_b = [x/metric_m_b for x in metric_m_b_arr] + \
            [metric_m_b/x for x in metric_m_b_arr]
        assert ratios_overall[mfn][('kk', 'aa')] == min(ratio_overall_k_a)
        assert ratios_overall[mfn][('kk', 'ba')] == min(ratio_overall_k_b)
        assert ratios_overall[mfn][('m', 'aa')] == min(ratio_overall_m_a)
        assert ratios_overall[mfn][('m', 'ba')] == pytest.approx(min(ratio_overall_m_b),
                                                                 rel=1e-10, abs=1e-16)
