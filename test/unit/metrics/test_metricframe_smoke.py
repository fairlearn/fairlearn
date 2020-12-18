# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

import fairlearn.metrics as metrics

from .data_for_test import y_t, y_p, g_1, g_2, g_3, g_4

from test.unit.input_convertors import conversions_for_1d


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
def test_basic(transform_y_t, transform_y_p):
    # If there are failures here, other, more specific tests should also fail
    g_f = pd.DataFrame(data=g_4, columns=['My feature'])
    target = metrics.MetricFrame(skm.recall_score,
                                 transform_y_t(y_t),
                                 transform_y_p(y_p),
                                 sensitive_features=g_f)

    # Check on the indices properties
    assert target.control_levels is None
    assert isinstance(target.sensitive_levels, list)
    assert (target.sensitive_levels == ['My feature'])

    # Check we have correct return types
    assert isinstance(target.overall, float)
    assert isinstance(target.by_group, pd.Series)

    # Check we have expected number of elements
    assert len(target.by_group) == 2
    assert np.array_equal(target.by_group.index.names, ['My feature'])

    recall_overall = skm.recall_score(y_t, y_p)
    assert target.overall == recall_overall

    mask_p = (g_4 == 'pp')
    mask_q = (g_4 == 'q')
    recall_p = skm.recall_score(y_t[mask_p], y_p[mask_p])
    recall_q = skm.recall_score(y_t[mask_q], y_p[mask_q])
    assert target.by_group['pp'] == recall_p
    assert target.by_group['q'] == recall_q

    target_mins = target.group_min()
    assert isinstance(target_mins, float)
    assert target_mins == min(recall_p, recall_q)

    target_maxes = target.group_max()
    assert isinstance(target_mins, float)
    assert target_maxes == max(recall_p, recall_q)


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
def test_basic_metric_dict(transform_y_t, transform_y_p):
    # If there are failures here, other, more specific tests should also fail
    g_f = pd.DataFrame(data=g_4, columns=['My feature'])
    target = metrics.MetricFrame({'recall': skm.recall_score},
                                 transform_y_t(y_t),
                                 transform_y_p(y_p),
                                 sensitive_features=g_f)

    # Check on the indices properties
    assert target.control_levels is None
    assert isinstance(target.sensitive_levels, list)
    assert (target.sensitive_levels == ['My feature'])

    # Check we have correct return types
    assert isinstance(target.overall, pd.Series)
    assert isinstance(target.by_group, pd.DataFrame)

    # Check we have expected number of elements
    assert len(target.overall) == 1
    assert target.by_group.shape == (2, 1)
    assert np.array_equal(target.by_group.index.names, ['My feature'])

    recall_overall = skm.recall_score(y_t, y_p)
    assert target.overall['recall'] == recall_overall

    mask_p = (g_4 == 'pp')
    mask_q = (g_4 == 'q')
    recall_p = skm.recall_score(y_t[mask_p], y_p[mask_p])
    recall_q = skm.recall_score(y_t[mask_q], y_p[mask_q])
    assert target.by_group['recall']['pp'] == recall_p
    assert target.by_group['recall']['q'] == recall_q

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.Series)
    assert len(target_mins) == 1
    assert target_mins['recall'] == min(recall_p, recall_q)

    target_maxes = target.group_max()
    assert isinstance(target_mins, pd.Series)
    assert len(target_maxes) == 1
    assert target_maxes['recall'] == max(recall_p, recall_q)


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
def test_1m_1sf_1cf(transform_y_t, transform_y_p):
    # If there are failures here, other, more specific tests should also fail
    target = metrics.MetricFrame(skm.recall_score,
                                 transform_y_t(y_t),
                                 transform_y_p(y_p),
                                 sensitive_features=g_2,
                                 control_features=g_3)

    # Check on the indices properties
    assert isinstance(target.control_levels, list)
    assert (target.control_levels == ['control_feature_0'])
    assert isinstance(target.sensitive_levels, list)
    assert (target.sensitive_levels == ['sensitive_feature_0'])

    # Check we have correct return types
    assert isinstance(target.overall, pd.Series)
    assert isinstance(target.by_group, pd.Series)

    mask_f = (g_2 == 'f')
    mask_g = (g_2 == 'g')
    mask_k = (g_3 == 'kk')
    mask_m = (g_3 == 'm')

    # Check we have expected number of elements
    assert len(target.overall) == 2
    assert len(target.by_group) == 4

    recall_k = skm.recall_score(y_t[mask_k], y_p[mask_k])
    recall_m = skm.recall_score(y_t[mask_m], y_p[mask_m])
    assert target.overall['kk'] == recall_k
    assert target.overall['m'] == recall_m

    mask_k_f = np.logical_and(mask_k, mask_f)
    mask_k_g = np.logical_and(mask_k, mask_g)
    mask_m_f = np.logical_and(mask_m, mask_f)
    mask_m_g = np.logical_and(mask_m, mask_g)
    recall_k_f = skm.recall_score(y_t[mask_k_f], y_p[mask_k_f])
    recall_m_f = skm.recall_score(y_t[mask_m_f], y_p[mask_m_f])
    recall_k_g = skm.recall_score(y_t[mask_k_g], y_p[mask_k_g])
    recall_m_g = skm.recall_score(y_t[mask_m_g], y_p[mask_m_g])
    assert target.by_group[('kk', 'f')] == recall_k_f
    assert target.by_group[('kk', 'g')] == recall_k_g
    assert target.by_group[('m', 'f')] == recall_m_f
    assert target.by_group[('m', 'g')] == recall_m_g

    recall_k_arr = [recall_k_f, recall_k_g]
    recall_m_arr = [recall_m_f, recall_m_g]

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.Series)
    assert len(target_mins) == 2
    assert target_mins['kk'] == min(recall_k_arr)
    assert target_mins['m'] == min(recall_m_arr)

    target_maxs = target.group_max()
    assert isinstance(target_mins, pd.Series)
    assert len(target_maxs) == 2
    assert target_maxs['kk'] == max(recall_k_arr)
    assert target_maxs['m'] == max(recall_m_arr)


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
def test_1m_1sf_1cf_metric_dict(transform_y_t, transform_y_p):
    # If there are failures here, other, more specific tests should also fail
    target = metrics.MetricFrame({'recall': skm.recall_score},
                                 transform_y_t(y_t),
                                 transform_y_p(y_p),
                                 sensitive_features=g_2,
                                 control_features=g_3)

    # Check on the indices properties
    assert isinstance(target.control_levels, list)
    assert (target.control_levels == ['control_feature_0'])
    assert isinstance(target.sensitive_levels, list)
    assert (target.sensitive_levels == ['sensitive_feature_0'])

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
    assert target.overall['recall']['kk'] == recall_k
    assert target.overall['recall']['m'] == recall_m

    mask_k_f = np.logical_and(mask_k, mask_f)
    mask_k_g = np.logical_and(mask_k, mask_g)
    mask_m_f = np.logical_and(mask_m, mask_f)
    mask_m_g = np.logical_and(mask_m, mask_g)
    recall_k_f = skm.recall_score(y_t[mask_k_f], y_p[mask_k_f])
    recall_m_f = skm.recall_score(y_t[mask_m_f], y_p[mask_m_f])
    recall_k_g = skm.recall_score(y_t[mask_k_g], y_p[mask_k_g])
    recall_m_g = skm.recall_score(y_t[mask_m_g], y_p[mask_m_g])
    assert target.by_group['recall'][('kk', 'f')] == recall_k_f
    assert target.by_group['recall'][('kk', 'g')] == recall_k_g
    assert target.by_group['recall'][('m', 'f')] == recall_m_f
    assert target.by_group['recall'][('m', 'g')] == recall_m_g

    recall_k_arr = [recall_k_f, recall_k_g]
    recall_m_arr = [recall_m_f, recall_m_g]

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_mins.shape == (2, 1)
    assert target_mins['recall']['kk'] == min(recall_k_arr)
    assert target_mins['recall']['m'] == min(recall_m_arr)

    target_maxs = target.group_max()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_maxs.shape == (2, 1)
    assert target_maxs['recall']['kk'] == max(recall_k_arr)
    assert target_maxs['recall']['m'] == max(recall_m_arr)


def test_duplicate_sf_names():
    groups = pd.DataFrame(np.stack([g_2, g_3], axis=1), columns=["A", "A"])
    msg = "Detected duplicate feature name: 'A'"
    with pytest.raises(ValueError) as execInfo:
        _ = metrics.MetricFrame(skm.recall_score,
                                y_t,
                                y_p,
                                sensitive_features=groups)
    assert execInfo.value.args[0] == msg


def test_duplicate_cf_names():
    groups = pd.DataFrame(np.stack([g_2, g_3], axis=1), columns=["B", "B"])
    msg = "Detected duplicate feature name: 'B'"
    with pytest.raises(ValueError) as execInfo:
        _ = metrics.MetricFrame(skm.recall_score,
                                y_t,
                                y_p,
                                sensitive_features=g_4,
                                control_features=groups)
    assert execInfo.value.args[0] == msg


def test_duplicate_cf_sf_names():
    cf = pd.DataFrame(np.stack([g_2, g_3], axis=1), columns=["A", "B"])
    sf = {"B": g_1, "C": g_4}
    msg = "Detected duplicate feature name: 'B'"
    with pytest.raises(ValueError) as execInfo:
        _ = metrics.MetricFrame(skm.recall_score,
                                y_t,
                                y_p,
                                sensitive_features=sf,
                                control_features=cf)
    assert execInfo.value.args[0] == msg


def test_single_element_lists():
    mf = metrics.MetricFrame(skm.balanced_accuracy_score,
                             [1], [1], sensitive_features=[0])
    assert mf.overall == 1
