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
def test_alternate_compute(transform_y_t, transform_y_p, num_batches):
    target = metrics.MetricFrame(skm.recall_score, [], [], sensitive_features=[],
                                 streaming=True)

    batch_generator = batchify(num_batches, y_t, y_p, g_4)
    for y_t_batches, y_p_batches, g_4_batches in batch_generator:
        g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
        target.add_data(transform_y_t(y_t_batches),
                        transform_y_p(y_p_batches),
                        sensitive_features=g_f)

    assert len(target._y_true) == num_batches
    assert target.overall == skm.recall_score(y_t, y_p)

    batch_generator = batchify(num_batches, y_t, y_p, g_4)
    for y_t_batches, y_p_batches, g_4_batches in batch_generator:
        g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
        target.add_data(transform_y_t(y_t_batches),
                        transform_y_p(y_p_batches),
                        sensitive_features=g_f)
    assert len(target._y_true) == 2 * num_batches
    assert target._overall is None  # has been reset
    assert target._by_group is None  # has been reset

    # We can recompute it with the new batches.
    assert len(target.by_group) == 2


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
@pytest.mark.parametrize("num_batches", [1, 3, 5])
def test_sample_params(transform_y_t, transform_y_p, num_batches):
    sample_weights = np.random.rand(len(y_t))
    batch_generator = batchify(num_batches, y_t, y_p, g_4, sample_weights)
    target = metrics.MetricFrame(skm.recall_score, [], [], sensitive_features=[],
                                 sample_params=None, streaming=True)

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

    # Compare to the non-batched version.
    g4 = pd.DataFrame(data=g_4, columns=['My feature'])
    met_regular = metrics.MetricFrame(skm.recall_score, y_t, y_p, sensitive_features=g4,
                                      sample_params={'sample_weight': sample_weights})
    assert target.overall == met_regular.overall

    for k, v in met_regular.by_group.items():
        assert v == target.by_group[k]

    assert target.group_min() == met_regular.group_min()
    assert target.group_max() == met_regular.group_max()

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

    # Compare to the non-batched version.
    g4 = pd.DataFrame(data=g_4, columns=['My feature'])
    met_regular = metrics.MetricFrame(skm.recall_score, y_t, y_p, sensitive_features=g4)
    assert target.overall == met_regular.overall

    for k, v in met_regular.by_group.items():
        assert v == target.by_group[k]

    assert target.group_min() == met_regular.group_min()
    assert target.group_max() == met_regular.group_max()

    # If we continue to add data, it resets the metrics
    g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
    target.add_data(transform_y_t(y_t_batches),
                    transform_y_p(y_p_batches),
                    sensitive_features=g_f)
    assert target._overall is None
    assert target._by_group is None


def test_exception_on_no_data():
    target = metrics.MetricFrame(skm.recall_score, [], [], sensitive_features=[],
                                 sample_params=None, streaming=True)

    with pytest.raises(ValueError) as excinfo:
        _ = target.control_levels
    assert "No data to process" in str(excinfo.value)
