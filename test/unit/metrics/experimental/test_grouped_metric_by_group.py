# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import sklearn.metrics as skm

from .data_for_test import y_t, y_p, s_w, g_1, g_2, g_3, g_4
from .utils import _get_raw_GroupedMetric


def test_1m_1sf_0cf():
    target = _get_raw_GroupedMetric()
    func_dict = target._process_functions(skm.recall_score, None, None)
    sf_list = target._process_features("SF", g_1, len(y_t))

    result = target._compute_by_group(func_dict, y_t, y_p, sf_list, None)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert np.array_equal(result.index.names, ['SF 0'])
    mask_a = (g_1 == 'aa')
    mask_b = (g_1 == 'ba')
    recall_a = skm.recall_score(y_t[mask_a], y_p[mask_a])
    recall_b = skm.recall_score(y_t[mask_b], y_p[mask_b])
    assert result['recall_score']['aa'] == recall_a
    assert result['recall_score']['ba'] == recall_b


def test_1m_1sf_1cf():
    target = _get_raw_GroupedMetric()
    func_dict = target._process_functions(skm.recall_score, None, None)
    sf_list = target._process_features("SF", g_1, len(y_t))
    cf_list = target._process_features("CF", g_2, len(y_t))

    result = target._compute_by_group(func_dict, y_t, y_p, sf_list, cf_list)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (4, 1)
    assert np.array_equal(result.index.names, ['CF 0', 'SF 0'])

    mask_a_f = np.logical_and((g_1 == 'aa'), (g_2 == 'f'))
    mask_a_g = np.logical_and((g_1 == 'aa'), (g_2 == 'g'))
    mask_b_f = np.logical_and((g_1 == 'ba'), (g_2 == 'f'))
    mask_b_g = np.logical_and((g_1 == 'ba'), (g_2 == 'g'))

    exp_a_f = skm.recall_score(y_t[mask_a_f], y_p[mask_a_f])
    exp_a_g = skm.recall_score(y_t[mask_a_g], y_p[mask_a_g])
    exp_b_f = skm.recall_score(y_t[mask_b_f], y_p[mask_b_f])
    exp_b_g = skm.recall_score(y_t[mask_b_g], y_p[mask_b_g])
    assert result['recall_score'][('f', 'aa')] == exp_a_f
    assert result['recall_score'][('f', 'ba')] == exp_b_f
    assert result['recall_score'][('g', 'aa')] == exp_a_g
    assert result['recall_score'][('g', 'ba')] == exp_b_g


def test_2m_2sf_2cf():
    target = _get_raw_GroupedMetric()
    func_dict = target._process_functions([skm.recall_score, skm.precision_score],
                                          [['sample_weight'], []],
                                          [{'sample_weight': s_w}, dict()])
    sf_list = target._process_features("Sens", [g_1, g_3], len(y_t))
    cf_list = target._process_features("Cond", [g_2, g_4], len(y_t))

    result = target._compute_by_group(func_dict, y_t, y_p, sf_list, cf_list)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (16, 2)
    assert np.array_equal(result.index.names, ['Cond 0', 'Cond 1', 'Sens 0', 'Sens 1'])

    # Only check some isolated results, rather than all 32
    mask_a_f = np.logical_and((g_1 == 'aa'), (g_2 == 'f'))
    mask_b_g = np.logical_and((g_1 == 'ba'), (g_2 == 'g'))
    mask_k_q = np.logical_and((g_3 == 'kk'), (g_4 == 'q'))

    mask_f_q_a_k = np.logical_and(mask_a_f, mask_k_q)
    recall_f_q_a_k = skm.recall_score(y_t[mask_f_q_a_k], y_p[mask_f_q_a_k],
                                      sample_weight=s_w[mask_f_q_a_k])
    prec_f_q_a_k = skm.precision_score(y_t[mask_f_q_a_k], y_p[mask_f_q_a_k])
    assert result['recall_score'][('f', 'q', 'aa', 'kk')] == recall_f_q_a_k
    assert result['precision_score'][('f', 'q', 'aa', 'kk')] == prec_f_q_a_k

    mask_g_q_b_k = np.logical_and(mask_b_g, mask_k_q)
    recall_g_q_b_k = skm.recall_score(y_t[mask_g_q_b_k], y_p[mask_g_q_b_k],
                                      sample_weight=s_w[mask_g_q_b_k])
    prec_g_q_b_k = skm.precision_score(y_t[mask_g_q_b_k], y_p[mask_g_q_b_k])
    assert result['recall_score'][('g', 'q', 'ba', 'kk')] == recall_g_q_b_k
    assert result['precision_score'][('g', 'q', 'ba', 'kk')] == prec_g_q_b_k
