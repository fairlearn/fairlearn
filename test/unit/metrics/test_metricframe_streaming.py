# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

import fairlearn.metrics as metrics
from test.unit.input_convertors import conversions_for_1d
from .data_for_test import y_t, y_p, g_4
from .utils import batchify


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
@pytest.mark.parametrize("num_batches", [1, 3, 5])
def test_sample_params(transform_y_t, transform_y_p, num_batches):
    sample_weights = np.random.rand(len(y_t))
    batch_generator = batchify(num_batches, y_t, y_p, g_4, sample_weights)
    target = metrics.MetricFrame(skm.recall_score, [], [], sensitive_features=[],
                                 sample_params=None, streaming=True)

    with pytest.raises(ValueError) as excinfo:
        _ = target.control_levels
    assert "No data to process" in str(excinfo.value)

    for y_t_batches, y_p_batches, g_4_batches, sw_batches in batch_generator:
        g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
        target.add_data(transform_y_t(y_t_batches),
                        transform_y_p(y_p_batches),
                        sensitive_features=g_f,
                        sample_params={'sample_weight': sw_batches})

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

    recall_overall = skm.recall_score(y_t, y_p, sample_weight=sample_weights)
    assert target.overall == recall_overall

    mask_p = (g_4 == 'pp')
    mask_q = (g_4 == 'q')
    recall_p = skm.recall_score(y_t[mask_p], y_p[mask_p], sample_weight=sample_weights[mask_p])
    recall_q = skm.recall_score(y_t[mask_q], y_p[mask_q], sample_weight=sample_weights[mask_q])
    assert target.by_group['pp'] == recall_p
    assert target.by_group['q'] == recall_q

    target_mins = target.group_min()
    assert isinstance(target_mins, float)
    assert target_mins == min(recall_p, recall_q)

    target_maxes = target.group_max()
    assert isinstance(target_mins, float)
    assert target_maxes == max(recall_p, recall_q)

    # If we continue to add data, it resets the metrics
    g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
    target.add_data(transform_y_t(y_t_batches),
                    transform_y_p(y_p_batches),
                    sensitive_features=g_f,
                    sample_params={'sample_weight': sw_batches})
    assert target._overall is None
    assert target._by_group is None



@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
@pytest.mark.parametrize("num_batches", [1, 3, 5])
def test_basic_streaming(transform_y_t, transform_y_p, num_batches):
    batch_generator = batchify(num_batches, y_t, y_p, g_4)
    target = metrics.MetricFrame(skm.recall_score, [], [], sensitive_features=[],
                                 streaming=True)

    with pytest.raises(ValueError) as excinfo:
        _ = target.control_levels
    assert "No data to process" in str(excinfo.value)

    for y_t_batches, y_p_batches, g_4_batches in batch_generator:
        g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
        target.add_data(transform_y_t(y_t_batches),
                        transform_y_p(y_p_batches),
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

    # If we continue to add data, it resets the metrics
    g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
    target.add_data(transform_y_t(y_t_batches),
                    transform_y_p(y_p_batches),
                    sensitive_features=g_f)
    assert target._overall is None
    assert target._by_group is None