# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

from fairlearn.metrics import MetricFrame

from .data_for_test import y_t, y_p, s_w, g_1, g_2, g_3, g_4


metric = [skm.recall_score,
          skm.precision_score,
          skm.accuracy_score,
          skm.balanced_accuracy_score]


@pytest.mark.parametrize("metric_fn", metric)
def test_1m_1sf_0cf(metric_fn):
    target = MetricFrame(metric_fn,
                         y_t, y_p,
                         sensitive_features=g_1)
    assert target._user_supplied_callable is True

    assert isinstance(target.by_group, pd.Series)
    assert len(target.by_group) == 2
    assert np.array_equal(target.by_group.index.names, ['sensitive_feature_0'])
    mask_a = (g_1 == 'aa')
    mask_b = (g_1 == 'ba')
    metric_a = metric_fn(y_t[mask_a], y_p[mask_a])
    metric_b = metric_fn(y_t[mask_b], y_p[mask_b])
    assert target.by_group['aa'] == metric_a
    assert target.by_group['ba'] == metric_b


@pytest.mark.parametrize("metric_fn", metric)
def test_1m_1sf_0cf_metric_dict(metric_fn):
    target = MetricFrame({metric_fn.__name__: metric_fn},
                         y_t, y_p,
                         sensitive_features=g_1)
    assert target._user_supplied_callable is False

    assert isinstance(target.by_group, pd.DataFrame)
    assert target.by_group.shape == (2, 1)
    assert np.array_equal(target.by_group.index.names, ['sensitive_feature_0'])
    mask_a = (g_1 == 'aa')
    mask_b = (g_1 == 'ba')
    metric_a = metric_fn(y_t[mask_a], y_p[mask_a])
    metric_b = metric_fn(y_t[mask_b], y_p[mask_b])
    assert target.by_group[metric_fn.__name__]['aa'] == metric_a
    assert target.by_group[metric_fn.__name__]['ba'] == metric_b


@ pytest.mark.parametrize("metric_fn", metric)
def test_1m_1sf_1cf(metric_fn):
    target = MetricFrame(metric_fn,
                         y_t, y_p,
                         sensitive_features=g_1,
                         control_features=g_2)
    assert target._user_supplied_callable is True

    assert isinstance(target.by_group, pd.Series)
    assert len(target.by_group) == 4
    assert np.array_equal(target.by_group.index.names,
                          ['control_feature_0', 'sensitive_feature_0'])

    mask_a_f = np.logical_and((g_1 == 'aa'), (g_2 == 'f'))
    mask_a_g = np.logical_and((g_1 == 'aa'), (g_2 == 'g'))
    mask_b_f = np.logical_and((g_1 == 'ba'), (g_2 == 'f'))
    mask_b_g = np.logical_and((g_1 == 'ba'), (g_2 == 'g'))

    exp_a_f = metric_fn(y_t[mask_a_f], y_p[mask_a_f])
    exp_a_g = metric_fn(y_t[mask_a_g], y_p[mask_a_g])
    exp_b_f = metric_fn(y_t[mask_b_f], y_p[mask_b_f])
    exp_b_g = metric_fn(y_t[mask_b_g], y_p[mask_b_g])
    assert target.by_group[('f', 'aa')] == exp_a_f
    assert target.by_group[('f', 'ba')] == exp_b_f
    assert target.by_group[('g', 'aa')] == exp_a_g
    assert target.by_group[('g', 'ba')] == exp_b_g


@ pytest.mark.parametrize("metric_fn", metric)
def test_1m_1sf_1cf_metric_dict(metric_fn):
    target = MetricFrame({metric_fn.__name__: metric_fn},
                         y_t, y_p,
                         sensitive_features=g_1,
                         control_features=g_2)
    assert target._user_supplied_callable is False

    assert isinstance(target.by_group, pd.DataFrame)
    assert target.by_group.shape == (4, 1)
    assert np.array_equal(target.by_group.index.names,
                          ['control_feature_0', 'sensitive_feature_0'])

    mask_a_f = np.logical_and((g_1 == 'aa'), (g_2 == 'f'))
    mask_a_g = np.logical_and((g_1 == 'aa'), (g_2 == 'g'))
    mask_b_f = np.logical_and((g_1 == 'ba'), (g_2 == 'f'))
    mask_b_g = np.logical_and((g_1 == 'ba'), (g_2 == 'g'))

    exp_a_f = metric_fn(y_t[mask_a_f], y_p[mask_a_f])
    exp_a_g = metric_fn(y_t[mask_a_g], y_p[mask_a_g])
    exp_b_f = metric_fn(y_t[mask_b_f], y_p[mask_b_f])
    exp_b_g = metric_fn(y_t[mask_b_g], y_p[mask_b_g])
    assert target.by_group[metric_fn.__name__][('f', 'aa')] == exp_a_f
    assert target.by_group[metric_fn.__name__][('f', 'ba')] == exp_b_f
    assert target.by_group[metric_fn.__name__][('g', 'aa')] == exp_a_g
    assert target.by_group[metric_fn.__name__][('g', 'ba')] == exp_b_g


def test_2m_2sf_2cf():
    funcs = {'recall': skm.recall_score, 'prec': skm.precision_score}
    s_p = {'recall': {'sample_weight': s_w}}
    sf = np.stack([g_1, g_3], axis=1)
    cf = {'cf1': g_2, 'cf2': g_4}

    target = MetricFrame(funcs,
                         y_t, y_p,
                         sensitive_features=sf,
                         control_features=cf,
                         sample_params=s_p)
    assert target._user_supplied_callable is False

    assert isinstance(target.by_group, pd.DataFrame)
    assert target.by_group.shape == (16, 2)
    assert np.array_equal(target.by_group.index.names,
                          ['cf1', 'cf2', 'sensitive_feature_0', 'sensitive_feature_1'])

    # Only check some isolated results, rather than all 32
    mask_a_f = np.logical_and((g_1 == 'aa'), (g_2 == 'f'))
    mask_b_g = np.logical_and((g_1 == 'ba'), (g_2 == 'g'))
    mask_k_q = np.logical_and((g_3 == 'kk'), (g_4 == 'q'))

    mask_f_q_a_k = np.logical_and(mask_a_f, mask_k_q)
    recall_f_q_a_k = skm.recall_score(y_t[mask_f_q_a_k], y_p[mask_f_q_a_k],
                                      sample_weight=s_w[mask_f_q_a_k])
    prec_f_q_a_k = skm.precision_score(y_t[mask_f_q_a_k], y_p[mask_f_q_a_k])
    assert target.by_group['recall'][('f', 'q', 'aa', 'kk')] == recall_f_q_a_k
    assert target.by_group['prec'][('f', 'q', 'aa', 'kk')] == prec_f_q_a_k

    mask_g_q_b_k = np.logical_and(mask_b_g, mask_k_q)
    recall_g_q_b_k = skm.recall_score(y_t[mask_g_q_b_k], y_p[mask_g_q_b_k],
                                      sample_weight=s_w[mask_g_q_b_k])
    prec_g_q_b_k = skm.precision_score(y_t[mask_g_q_b_k], y_p[mask_g_q_b_k])
    assert target.by_group['recall'][('g', 'q', 'ba', 'kk')] == recall_g_q_b_k
    assert target.by_group['prec'][('g', 'q', 'ba', 'kk')] == prec_g_q_b_k
