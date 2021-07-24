# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

import fairlearn.metrics as metrics
from test.unit.input_convertors import (conversions_for_1d,
                                        ensure_ndarray,
                                        ensure_list,
                                        ensure_dataframe)
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
        target.add_data(y_true=transform_y_t(y_t_batches),
                        y_pred=transform_y_p(y_p_batches),
                        sensitive_features=g_f)

    assert len(target._y_true) == num_batches
    assert target.overall == skm.recall_score(y_t, y_p)

    batch_generator = batchify(num_batches, y_t, y_p, g_4)
    for y_t_batches, y_p_batches, g_4_batches in batch_generator:
        g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
        target.add_data(y_true=transform_y_t(y_t_batches),
                        y_pred=transform_y_p(y_p_batches),
                        sensitive_features=g_f)
    assert len(target._y_true) == 2 * num_batches
    assert target._overall is None  # has been reset
    assert target._by_group is None  # has been reset

    # We check that calling `by_group` will recompute the values.
    assert len(target.by_group) == 2


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
@pytest.mark.parametrize("num_batches", [1, 3, 5])
def test_sample_params(transform_y_t, transform_y_p, num_batches):
    sample_weights = np.random.rand(len(y_t))
    batch_generator = batchify(num_batches, y_t, y_p, g_4, sample_weights)
    target = metrics.MetricFrame(skm.recall_score, [], [], sensitive_features=[],
                                 sample_params={}, streaming=True)

    for y_t_batches, y_p_batches, g_4_batches, sw_batches in batch_generator:
        g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
        target.add_data(y_true=transform_y_t(y_t_batches),
                        y_pred=transform_y_p(y_p_batches),
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
    met_regular = metrics.MetricFrame(metrics=skm.recall_score, y_true=y_t,
                                      y_pred=y_p, sensitive_features=g4,
                                      sample_params={'sample_weight': sample_weights})
    assert target.overall == met_regular.overall

    for k, v in met_regular.by_group.items():
        assert v == target.by_group[k]

    assert target.group_min() == met_regular.group_min()
    assert target.group_max() == met_regular.group_max()

    # If we continue to add data, it resets the metrics
    g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
    target.add_data(y_true=transform_y_t(y_t_batches),
                    y_pred=transform_y_p(y_p_batches),
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
        target.add_data(y_true=transform_y_t(y_t_batches),
                        y_pred=transform_y_p(y_p_batches),
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
    met_regular = metrics.MetricFrame(metrics=skm.recall_score,
                                      y_true=y_t, y_pred=y_p,
                                      sensitive_features=g4)
    assert target.overall == met_regular.overall

    for k, v in met_regular.by_group.items():
        assert v == target.by_group[k]

    assert target.group_min() == met_regular.group_min()
    assert target.group_max() == met_regular.group_max()

    # If we continue to add data, it resets the metrics
    g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
    target.add_data(y_true=transform_y_t(y_t_batches),
                    y_pred=transform_y_p(y_p_batches),
                    sensitive_features=g_f)
    assert target._overall is None
    assert target._by_group is None


def test_metric_frame_wrong_initialization():
    g4 = pd.DataFrame(data=g_4, columns=['My feature'])
    # We can't initialize streaming metrics with data
    with pytest.raises(ValueError) as excinfo:
        _ = metrics.MetricFrame(metrics=skm.recall_score, y_true=y_t, y_pred=y_p,
                                sensitive_features=g4, control_features=g4,
                                streaming=True)
    assert "Streaming metrics must be initialized with empty lists" in str(excinfo.value)


def test_exception_on_no_data():
    target = metrics.MetricFrame(skm.recall_score, [], [], sensitive_features=[],
                                 sample_params=None, streaming=True)

    with pytest.raises(ValueError) as excinfo:
        _ = target.control_levels
    assert "No data to process" in str(excinfo.value)


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
@pytest.mark.parametrize("num_batches", [1, 3, 5])
def test_cf_argument_mismatch(transform_y_t, transform_y_p, num_batches):
    batch_generator = batchify(num_batches, y_t, y_p, g_4)

    # We initialize without CF
    target = metrics.MetricFrame(skm.recall_score, [], [], sensitive_features=[],
                                 streaming=True)
    # We supply CF in the call, it should fail.
    with pytest.raises(ValueError) as excinfo:
        y_t_batches, y_p_batches, g_4_batches = next(batch_generator)
        g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
        target.add_data(y_true=transform_y_t(y_t_batches),
                        y_pred=transform_y_p(y_p_batches),
                        sensitive_features=g_f,
                        control_features=g_f)
    assert "MetricFrame expected `control_features=None`" in str(excinfo.value)


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
@pytest.mark.parametrize("num_batches", [1, 3, 5])
def test_cf_argument_success(transform_y_t, transform_y_p, num_batches):
    batch_generator = batchify(num_batches, y_t, y_p, g_4)
    # We initialize with CF
    target = metrics.MetricFrame(skm.recall_score, [], [], sensitive_features=[],
                                 control_features=[], streaming=True)
    # We supply CF in the call.
    y_t_batches, y_p_batches, g_4_batches = next(batch_generator)
    g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
    target.add_data(y_true=transform_y_t(y_t_batches),
                    y_pred=transform_y_p(y_p_batches),
                    sensitive_features=g_f,
                    control_features=g_f)


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
def test_sp_argument_mismatch(transform_y_t, transform_y_p):
    num_batches = 2
    sample_weights = np.random.rand(len(y_t))
    batch_generator = batchify(num_batches, y_t, y_p, g_4, sample_weights)
    target = metrics.MetricFrame(skm.recall_score, [], [], sensitive_features=[],
                                 sample_params={}, streaming=True)

    y_t_batches, y_p_batches, g_4_batches, sw_batches = next(batch_generator)
    g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
    target.add_data(y_true=transform_y_t(y_t_batches),
                    y_pred=transform_y_p(y_p_batches),
                    sensitive_features=g_f,
                    sample_params={'sample_weight': sw_batches})

    # We forget to supply it the next time, it should raise.
    with pytest.raises(ValueError) as excinfo:
        target.add_data(y_true=transform_y_t(y_t_batches),
                        y_pred=transform_y_p(y_p_batches),
                        sensitive_features=g_f)
    assert "MetricFrame expected `sample_params` to be supplied" in str(excinfo.value)


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
def test_concat_batches(transform_y_p):
    num_batches = 5
    metric_frame = metrics.MetricFrame(skm.recall_score, [], [], sensitive_features=[],
                                       sample_params={}, streaming=True)
    # Get y_p from the batch to make a list of batch.
    batches = [b for [b] in batchify(num_batches, transform_y_p(y_p))]
    concatenated = metric_frame._concat_batches(batches)
    assert len(concatenated) == len(y_p)


def test_concat_batches_error():
    metric_frame = metrics.MetricFrame(skm.recall_score, [], [], sensitive_features=[],
                                       sample_params={}, streaming=True)
    with pytest.raises(ValueError) as excinfo:
        metric_frame._concat_batches([])
    assert "No data to process" in str(excinfo.value)

    lst_input, arr_input = ensure_list(y_t), ensure_ndarray(y_t)
    with pytest.raises(ValueError) as excinfo:
        metric_frame._concat_batches([lst_input, arr_input])
    assert "Can't concatenate arrays of different types" in str(excinfo.value)

    df1, df2 = ensure_dataframe(y_t), ensure_dataframe(y_t)
    # Edit column name
    df1.columns = [f"{c}_" for c in df1.columns]
    with pytest.raises(ValueError) as excinfo:
        metric_frame._concat_batches([df1, df2])
    assert "Column mismatch" in str(excinfo.value)
