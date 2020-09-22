# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import sklearn.metrics as skm

from .data_for_test import y_t, y_p, s_w, g_1, g_2
from .utils import _get_raw_GroupedMetric


def test_1m_0cf():
    target = _get_raw_GroupedMetric()
    func_dict = target._process_functions(skm.recall_score, None, None)
    result = target._compute_overall(func_dict, y_t, y_p, None)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 1)
    expected = skm.recall_score(y_t, y_p)
    assert result['recall_score']['overall'] == expected


def test_1m_0cf_wgt():
    target = _get_raw_GroupedMetric()
    func_dict = target._process_functions(skm.recall_score,
                                          ['sample_weight'],
                                          {'sample_weight': s_w})

    result = target._compute_overall(func_dict, y_t, y_p, None)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 1)
    expected = skm.recall_score(y_t, y_p, sample_weight=s_w)
    assert result['recall_score']['overall'] == expected


def test_2m_0cf():
    target = _get_raw_GroupedMetric()
    func_dict = target._process_functions([skm.recall_score, skm.precision_score],
                                          None, None)
    result = target._compute_overall(func_dict, y_t, y_p, None)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 2)
    exp_recall = skm.recall_score(y_t, y_p)
    exp_prec = skm.precision_score(y_t, y_p)
    assert result['recall_score']['overall'] == exp_recall
    assert result['precision_score']['overall'] == exp_prec


def test_1m_1cf():
    target = _get_raw_GroupedMetric()
    func_dict = target._process_functions(skm.recall_score, None, None)
    cf_list = target._process_features("CF", g_1, len(y_t))

    result = target._compute_overall(func_dict, y_t, y_p, cf_list)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    mask_a = (g_1 == 'a')
    mask_b = (g_1 == 'b')
    exp_recall_a = skm.recall_score(y_t[mask_a], y_p[mask_a])
    exp_recall_b = skm.recall_score(y_t[mask_b], y_p[mask_b])
    assert result['recall_score'][('a',)] == exp_recall_a
    assert result['recall_score'][('b',)] == exp_recall_b


def test_1m_1cf_wgt():
    target = _get_raw_GroupedMetric()
    func_dict = target._process_functions(skm.recall_score,
                                          ['sample_weight'],
                                          {'sample_weight': s_w})
    cf_list = target._process_features("CF", g_1, len(y_t))

    result = target._compute_overall(func_dict, y_t, y_p, cf_list)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    mask_a = (g_1 == 'a')
    mask_b = (g_1 == 'b')
    exp_recall_a = skm.recall_score(y_t[mask_a], y_p[mask_a],
                                    sample_weight=s_w[mask_a])
    exp_recall_b = skm.recall_score(y_t[mask_b], y_p[mask_b],
                                    sample_weight=s_w[mask_b])
    assert result['recall_score'][('a',)] == exp_recall_a
    assert result['recall_score'][('b',)] == exp_recall_b


def test_1m_2cf():
    target = _get_raw_GroupedMetric()
    func_dict = target._process_functions(skm.recall_score, None, None)
    cf_list = target._process_features("CF", [g_1, g_2], len(y_t))

    result = target._compute_overall(func_dict, y_t, y_p, cf_list)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (4, 1)
    mask_a_f = np.logical_and((g_1 == 'a'), (g_2 == 'f'))
    mask_a_g = np.logical_and((g_1 == 'a'), (g_2 == 'g'))
    mask_b_f = np.logical_and((g_1 == 'b'), (g_2 == 'f'))
    mask_b_g = np.logical_and((g_1 == 'b'), (g_2 == 'g'))

    exp_a_f = skm.recall_score(y_t[mask_a_f], y_p[mask_a_f])
    exp_a_g = skm.recall_score(y_t[mask_a_g], y_p[mask_a_g])
    exp_b_f = skm.recall_score(y_t[mask_b_f], y_p[mask_b_f])
    exp_b_g = skm.recall_score(y_t[mask_b_g], y_p[mask_b_g])

    assert result['recall_score'][('a', 'f')] == exp_a_f
    assert result['recall_score'][('a', 'g')] == exp_a_g
    assert result['recall_score'][('b', 'f')] == exp_b_f
    assert result['recall_score'][('b', 'g')] == exp_b_g


def test_2m_2cf():
    target = _get_raw_GroupedMetric()
    func_dict = target._process_functions([skm.recall_score, skm.precision_score], None, None)
    cf_list = target._process_features("CF", [g_1, g_2], len(y_t))

    result = target._compute_overall(func_dict, y_t, y_p, cf_list)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (4, 2)
    mask_a_f = np.logical_and((g_1 == 'a'), (g_2 == 'f'))
    mask_a_g = np.logical_and((g_1 == 'a'), (g_2 == 'g'))
    mask_b_f = np.logical_and((g_1 == 'b'), (g_2 == 'f'))
    mask_b_g = np.logical_and((g_1 == 'b'), (g_2 == 'g'))

    recall_a_f = skm.recall_score(y_t[mask_a_f], y_p[mask_a_f])
    recall_a_g = skm.recall_score(y_t[mask_a_g], y_p[mask_a_g])
    recall_b_f = skm.recall_score(y_t[mask_b_f], y_p[mask_b_f])
    recall_b_g = skm.recall_score(y_t[mask_b_g], y_p[mask_b_g])
    assert result['recall_score'][('a', 'f')] == recall_a_f
    assert result['recall_score'][('a', 'g')] == recall_a_g
    assert result['recall_score'][('b', 'f')] == recall_b_f
    assert result['recall_score'][('b', 'g')] == recall_b_g

    prec_a_f = skm.precision_score(y_t[mask_a_f], y_p[mask_a_f])
    prec_a_g = skm.precision_score(y_t[mask_a_g], y_p[mask_a_g])
    prec_b_f = skm.precision_score(y_t[mask_b_f], y_p[mask_b_f])
    prec_b_g = skm.precision_score(y_t[mask_b_g], y_p[mask_b_g])
    assert result['precision_score'][('a', 'f')] == prec_a_f
    assert result['precision_score'][('a', 'g')] == prec_a_g
    assert result['precision_score'][('b', 'f')] == prec_b_f
    assert result['precision_score'][('b', 'g')] == prec_b_g
