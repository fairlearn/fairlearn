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
@pytest.mark.parametrize("num_batches", [1, 3, 5, len(y_t)])
def test_alternate_compute(transform_y_t, transform_y_p, num_batches):
    target = metrics.StreamingMetricFrame(metrics=skm.recall_score)

    batch_generator = batchify(num_batches, y_t, y_p, g_4)
    for y_t_batches, y_p_batches, g_4_batches in batch_generator:
        g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
        target.add_data(y_true=transform_y_t(y_t_batches),
                        y_pred=transform_y_p(y_p_batches),
                        sensitive_features=g_f)

    assert len(target._y_true) == len(y_t)
    assert target.get_metric_frame().overall == skm.recall_score(y_t, y_p)

    batch_generator = batchify(num_batches, y_t, y_p, g_4)
    for y_t_batches, y_p_batches, g_4_batches in batch_generator:
        g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
        target.add_data(y_true=transform_y_t(y_t_batches),
                        y_pred=transform_y_p(y_p_batches),
                        sensitive_features=g_f)
    assert len(target._y_true) == 2 * len(y_t)

    # We check that calling `by_group` will recompute the values.
    assert len(target.get_metric_frame().by_group) == 2


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
@pytest.mark.parametrize("num_batches", [1, 3, 5, len(y_t)])
def test_simple_sample_params(transform_y_t, transform_y_p, num_batches):
    sample_weights = np.random.rand(len(y_t))
    batch_generator = batchify(num_batches, y_t, y_p, g_4, sample_weights)
    target = metrics.StreamingMetricFrame(metrics=skm.recall_score)

    for y_t_batches, y_p_batches, g_4_batches, sw_batches in batch_generator:
        g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
        target.add_data(y_true=transform_y_t(y_t_batches),
                        y_pred=transform_y_p(y_p_batches),
                        sensitive_features=g_f,
                        sample_params={'sample_weight': sw_batches})

    target_metric = target.get_metric_frame()
    # Check we have correct return types
    assert isinstance(target_metric.overall, float)
    assert isinstance(target_metric.by_group, pd.Series)

    # Check we have expected number of elements
    assert len(target_metric.by_group) == 2
    assert np.array_equal(target_metric.by_group.index.names, ['My feature'])

    # Compare to the non-batched version.
    g4 = pd.DataFrame(data=g_4, columns=['My feature'])
    met_regular = metrics.MetricFrame(metrics=skm.recall_score, y_true=y_t,
                                      y_pred=y_p, sensitive_features=g4,
                                      sample_params={'sample_weight': sample_weights})
    assert target_metric.overall == met_regular.overall

    for k, v in met_regular.by_group.items():
        assert v == target_metric.by_group[k]

    assert target_metric.group_min() == met_regular.group_min()
    assert target_metric.group_max() == met_regular.group_max()


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
@pytest.mark.parametrize("num_batches", [1, 3, 5, len(y_t)])
def test_basic_streaming(transform_y_t, transform_y_p, num_batches):
    batch_generator = batchify(num_batches, y_t, y_p, g_4)
    target = metrics.StreamingMetricFrame(metrics=skm.recall_score)

    for y_t_batches, y_p_batches, g_4_batches in batch_generator:
        g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
        target.add_data(y_true=transform_y_t(y_t_batches),
                        y_pred=transform_y_p(y_p_batches),
                        sensitive_features=g_f)

    target_metric = target.get_metric_frame()
    # Check on the indices properties
    assert target_metric.control_levels is None
    assert isinstance(target_metric.sensitive_levels, list)
    assert (target_metric.sensitive_levels == ['My feature'])

    # Check we have correct return types
    assert isinstance(target_metric.overall, float)
    assert isinstance(target_metric.by_group, pd.Series)

    # Check we have expected number of elements
    assert len(target_metric.by_group) == 2
    assert np.array_equal(target_metric.by_group.index.names, ['My feature'])

    # Compare to the non-batched version.
    g4 = pd.DataFrame(data=g_4, columns=['My feature'])
    met_regular = metrics.MetricFrame(metrics=skm.recall_score,
                                      y_true=y_t, y_pred=y_p,
                                      sensitive_features=g4)
    assert target_metric.overall == met_regular.overall

    for k, v in met_regular.by_group.items():
        assert v == target_metric.by_group[k]

    assert target_metric.group_min() == met_regular.group_min()
    assert target_metric.group_max() == met_regular.group_max()

    # Verify that reset works.
    target.reset()
    assert all(v is None for v in (target._y_true, target._y_pred, target._sensitive_features,
                                   target._control_features, target._sample_params))


def test_exception_on_no_data():
    target = metrics.StreamingMetricFrame(metrics=skm.recall_score)

    with pytest.raises(ValueError) as excinfo:
        _ = target.get_metric_frame()
    assert "No data to process" in str(excinfo.value)


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
def test_concat_batches(transform_y_p):
    num_batches = 5
    metric_frame = metrics.StreamingMetricFrame(metrics=skm.recall_score)
    # Get y_p from the batch to make a list of batch.
    batches = [b for [b] in batchify(num_batches, transform_y_p(y_p))]
    concatenated = metric_frame._concat_batches(batches)
    assert len(concatenated) == len(y_p)


def test_concat_batches_error():
    metric_frame = metrics.StreamingMetricFrame(metrics=skm.recall_score)
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


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
@pytest.mark.parametrize("num_batches", [1, 3, 5])
def test_multimetric_straming(transform_y_t, transform_y_p, num_batches):
    batch_generator = batchify(num_batches, y_t, y_p, g_4)
    target = metrics.StreamingMetricFrame(metrics={'recall': skm.recall_score,
                                                   'precision': skm.precision_score})

    for y_t_batches, y_p_batches, g_4_batches in batch_generator:
        g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
        target.add_data(y_true=transform_y_t(y_t_batches),
                        y_pred=transform_y_p(y_p_batches),
                        sensitive_features=g_f)

    target_metric = target.get_metric_frame()

    # Compare to the non-batched version.
    g4 = pd.DataFrame(data=g_4, columns=['My feature'])
    met_regular = metrics.MetricFrame(metrics={'recall': skm.recall_score,
                                               'precision': skm.precision_score},
                                      y_true=y_t, y_pred=y_p,
                                      sensitive_features=g4)
    assert (target_metric.overall == met_regular.overall).all()

    for k, v in met_regular.by_group.items():
        assert (v == target_metric.by_group[k]).all()

    assert (target_metric.group_min() == met_regular.group_min()).all()
    assert (target_metric.group_max() == met_regular.group_max()).all()


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
@pytest.mark.parametrize("num_batches", [1, 3, 5])
def test_nested_sample_params(transform_y_t, transform_y_p, num_batches):
    sample_weights = np.random.rand(len(y_t))
    batch_generator = batchify(num_batches, y_t, y_p, g_4, sample_weights)
    target = metrics.StreamingMetricFrame(metrics={'recall': skm.recall_score,
                                                   'precision': skm.precision_score})

    for y_t_batches, y_p_batches, g_4_batches, sw_batches in batch_generator:
        g_f = pd.DataFrame(data=g_4_batches, columns=['My feature'])
        target.add_data(y_true=transform_y_t(y_t_batches),
                        y_pred=transform_y_p(y_p_batches),
                        sensitive_features=g_f,
                        sample_params={'recall': {'sample_weight': sw_batches},
                                       'precision': {'sample_weight': sw_batches}})

    target_metric = target.get_metric_frame()

    # Compare to the non-batched version.
    g4 = pd.DataFrame(data=g_4, columns=['My feature'])
    weights_ = {'recall': {'sample_weight': sample_weights},
                'precision': {'sample_weight': sample_weights}}
    met_regular = metrics.MetricFrame(metrics={'recall': skm.recall_score,
                                               'precision': skm.precision_score}, y_true=y_t,
                                      y_pred=y_p, sensitive_features=g4,
                                      sample_params=weights_)
    assert (target_metric.overall == met_regular.overall).all()

    for k, v in met_regular.by_group.items():
        assert (v == target_metric.by_group[k]).all()

    assert (target_metric.group_min() == met_regular.group_min()).all()
    assert (target_metric.group_max() == met_regular.group_max()).all()


def test_batchsize_mismatch():
    target = metrics.StreamingMetricFrame(metrics=skm.recall_score)

    batch_2 = y_t[:-1], y_p, g_4
    batch_3 = y_t, y_p[:-1], g_4
    batch_4 = y_t, y_p, g_4[:-1]

    for y_true, y_pred, sensitive in (batch_2, batch_3, batch_4):
        target.reset()
        target.add_data(y_true=y_t, y_pred=y_p, sensitive_features=g_4)
        with pytest.raises(ValueError) as excinfo:
            target.add_data(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive)
        assert "inconsistent numbers of samples" in str(excinfo.value)
