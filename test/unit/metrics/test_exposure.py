# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import pytest
import numpy as np

from fairlearn.metrics import MetricFrame
from fairlearn.metrics import exposure, utility, proportional_exposure
from fairlearn.metrics import (
    exposure_difference,
    exposure_ratio,
    proportional_exposure_difference,
    proportional_exposure_ratio
)
from fairlearn.metrics._exposure import (
    _EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE,
    _INVALID_RANKING_ERROR_MESSAGE
)

from .data_for_test import y_t, s_w, g_1

y_p = list(range(len(y_t)))

_aggregate_methods = ['between_groups', 'to_overall']

'''
    Test metrics
'''


def test_exposure_empty():
    with pytest.raises(ValueError) as exc:
        _ = exposure([], [])
    assert _EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE.format('y_pred') == exc.value.args[0]


def test_utility_empty():
    with pytest.raises(ValueError) as exc:
        _ = utility([], [])
    assert _EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE.format('y_true') == exc.value.args[0]


def test_proportional_exposure_empty():
    with pytest.raises(ValueError) as exc:
        _ = proportional_exposure([], [])
    assert _EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE.format('y_pred') == exc.value.args[0]

    with pytest.raises(ValueError) as exc:
        _ = proportional_exposure([], [1, 2, 3])
    assert _EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE.format('y_true') == exc.value.args[0]


def test_exposure_invalid_ranking():
    with pytest.raises(ValueError) as exc:
        _ = exposure([1, 1, 1], [1, 1, 1])
    assert _INVALID_RANKING_ERROR_MESSAGE == exc.value.args[0]


def test_proportional_exposure_invalid_ranking():
    with pytest.raises(ValueError) as exc:
        _ = proportional_exposure([1, 1, 1], [1, 1, 1])
    assert _INVALID_RANKING_ERROR_MESSAGE == exc.value.args[0]


def test_exposure_single_element():
    assert exposure([], [1]) == 1
    assert exposure([], [10]) == 1/np.log2(1 + 10)


def test_utility_single_element():
    assert utility([1], []) == 1
    assert utility([1, 2], []) == 1.5
    assert utility([1, 2, 3], []) == 2


def test_proportional_exposure_single_element():
    assert proportional_exposure([1], [1]) == 1
    assert proportional_exposure([10], [1]) == 0.1


def test_exposure_unweighted():
    result = exposure(y_t, y_p)
    assert result == 0.23141957054179535


def test_exposure_weighted():
    result = exposure(y_t, y_p, sample_weight=s_w)
    assert result == 0.4384130413271165


def test_utility_unweighted():
    result = utility(y_t, y_p)
    assert result == 0.6338028169014085


def test_utility_weighted():
    result = utility(y_t, y_p, sample_weight=s_w)
    assert result == 1.323943661971831


def test_proportional_exposure_unweighted():
    result = proportional_exposure(y_t, y_p)
    assert result == 0.36512865574372155


def test_proportional_exposure_weighted():
    result = proportional_exposure(y_t, y_p, sample_weight=s_w)
    assert result == 0.33114176525771566


'''
    Test disparities
'''


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_exposure_difference(agg_method):
    actual = exposure_difference(y_t, y_p, sensitive_features=g_1, method=agg_method)

    gm = MetricFrame(metrics=exposure, y_true=y_t, y_pred=y_p, sensitive_features=g_1)

    assert actual == gm.difference(method=agg_method)


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_exposure_difference_weighted(agg_method):
    actual = exposure_difference(y_t, y_p,
                                 sensitive_features=g_1,
                                 sample_weight=s_w,
                                 method=agg_method)

    gm = MetricFrame(metrics=exposure, y_true=y_t, y_pred=y_p,
                     sensitive_features=g_1,
                     sample_params={'sample_weight': s_w})

    assert actual == gm.difference(method=agg_method)


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_exposure_ratio(agg_method):
    actual = exposure_ratio(y_t, y_p, sensitive_features=g_1, method=agg_method)

    gm = MetricFrame(metrics=exposure, y_true=y_t, y_pred=y_p, sensitive_features=g_1)

    assert actual == gm.ratio(method=agg_method)


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_exposure_ratio_weighted(agg_method):
    actual = exposure_ratio(y_t, y_p,
                            sensitive_features=g_1,
                            sample_weight=s_w,
                            method=agg_method)

    gm = MetricFrame(metrics=exposure, y_true=y_t, y_pred=y_p,
                     sensitive_features=g_1,
                     sample_params={'sample_weight': s_w})

    assert actual == gm.ratio(method=agg_method)


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_proportional_exposure_difference(agg_method):
    actual = proportional_exposure_difference(y_t, y_p, sensitive_features=g_1, method=agg_method)

    gm = MetricFrame(metrics=proportional_exposure, y_true=y_t, y_pred=y_p, sensitive_features=g_1)

    assert actual == gm.difference(method=agg_method)


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_proportional_exposure_difference_weighted(agg_method):
    actual = proportional_exposure_difference(y_t, y_p,
                                              sensitive_features=g_1,
                                              sample_weight=s_w,
                                              method=agg_method)

    gm = MetricFrame(metrics=proportional_exposure, y_true=y_t, y_pred=y_p,
                     sensitive_features=g_1,
                     sample_params={'sample_weight': s_w})

    assert actual == gm.difference(method=agg_method)


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_proportional_exposure_ratio(agg_method):
    actual = proportional_exposure_ratio(y_t, y_p, sensitive_features=g_1, method=agg_method)

    gm = MetricFrame(metrics=proportional_exposure, y_true=y_t, y_pred=y_p, sensitive_features=g_1)

    assert actual == gm.ratio(method=agg_method)


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_proportional_exposure_ratio_weighted(agg_method):
    actual = proportional_exposure_ratio(y_t, y_p,
                                         sensitive_features=g_1,
                                         sample_weight=s_w,
                                         method=agg_method)

    gm = MetricFrame(metrics=proportional_exposure, y_true=y_t, y_pred=y_p,
                     sensitive_features=g_1,
                     sample_params={'sample_weight': s_w})

    assert actual == gm.ratio(method=agg_method)
