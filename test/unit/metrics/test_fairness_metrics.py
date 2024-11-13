# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from itertools import product

import pytest

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equal_opportunity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)

from .data_for_test import g_1, s_w, y_p, y_t

_aggregate_methods = ["between_groups", "to_overall"]
_agg_options = ["worst_case", "mean"]
# cartesian product of the two lists
_agg_combinations = list(product(_aggregate_methods, _agg_options))


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_demographic_parity_difference(agg_method):
    actual = demographic_parity_difference(y_t, y_p, sensitive_features=g_1, method=agg_method)

    gm = MetricFrame(metrics=selection_rate, y_true=y_t, y_pred=y_p, sensitive_features=g_1)

    assert actual == gm.difference(method=agg_method)


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_demographic_parity_difference_weighted(agg_method):
    actual = demographic_parity_difference(
        y_t, y_p, sensitive_features=g_1, sample_weight=s_w, method=agg_method
    )

    gm = MetricFrame(
        metrics=selection_rate,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        sample_params={"sample_weight": s_w},
    )

    assert actual == gm.difference(method=agg_method)


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_demographic_parity_ratio(agg_method):
    actual = demographic_parity_ratio(y_t, y_p, sensitive_features=g_1, method=agg_method)

    gm = MetricFrame(metrics=selection_rate, y_true=y_t, y_pred=y_p, sensitive_features=g_1)

    assert actual == gm.ratio(method=agg_method)


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_demographic_parity_ratio_weighted(agg_method):
    actual = demographic_parity_ratio(
        y_t, y_p, sensitive_features=g_1, sample_weight=s_w, method=agg_method
    )

    gm = MetricFrame(
        metrics=selection_rate,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        sample_params={"sample_weight": s_w},
    )

    assert actual == gm.ratio(method=agg_method)


@pytest.mark.parametrize("agg_method, agg", _agg_combinations)
def test_equalized_odds_difference(agg_method, agg):
    actual = equalized_odds_difference(
        y_t, y_p, sensitive_features=g_1, method=agg_method, agg=agg
    )

    metrics = {"tpr": true_positive_rate, "fpr": false_positive_rate}
    gm = MetricFrame(metrics=metrics, y_true=y_t, y_pred=y_p, sensitive_features=g_1)

    diffs = gm.difference(method=agg_method)
    if agg == "worst_case":
        assert actual == diffs.max()
    else:
        assert actual == diffs.mean()


@pytest.mark.parametrize("agg_method, agg", _agg_combinations)
def test_equalized_odds_difference_weighted(agg_method, agg):
    actual = equalized_odds_difference(
        y_t, y_p, sensitive_features=g_1, method=agg_method, sample_weight=s_w, agg=agg
    )

    metrics = {"tpr": true_positive_rate, "fpr": false_positive_rate}
    sw = {"sample_weight": s_w}
    sp = {"tpr": sw, "fpr": sw}
    gm = MetricFrame(
        metrics=metrics,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        sample_params=sp,
    )

    diffs = gm.difference(method=agg_method)
    if agg == "worst_case":
        assert actual == diffs.max()
    else:
        assert actual == diffs.mean()


@pytest.mark.parametrize("agg_method, agg", _agg_combinations)
def test_equalized_odds_ratio(agg_method, agg):
    actual = equalized_odds_ratio(y_t, y_p, method=agg_method, sensitive_features=g_1, agg=agg)

    metrics = {"tpr": true_positive_rate, "fpr": false_positive_rate}
    gm = MetricFrame(metrics=metrics, y_true=y_t, y_pred=y_p, sensitive_features=g_1)

    ratios = gm.ratio(method=agg_method)
    if agg == "worst_case":
        assert actual == ratios.min()
    else:
        assert actual == ratios.mean()


@pytest.mark.parametrize("agg_method, agg", _agg_combinations)
def test_equalized_odds_ratio_weighted(agg_method, agg):
    actual = equalized_odds_ratio(
        y_t, y_p, method=agg_method, sensitive_features=g_1, sample_weight=s_w, agg=agg
    )

    metrics = {"tpr": true_positive_rate, "fpr": false_positive_rate}
    sw = {"sample_weight": s_w}
    sp = {"tpr": sw, "fpr": sw}
    gm = MetricFrame(
        metrics=metrics,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        sample_params=sp,
    )

    ratios = gm.ratio(method=agg_method)
    if agg == "worst_case":
        assert actual == ratios.min()
    else:
        assert actual == ratios.mean()


@pytest.mark.parametrize(
    ["agg", "expectation"],
    [
        (
            "wrong-arg",
            pytest.raises(
                ValueError, match="agg must be one of 'worst_case' or 'mean', got wrong-arg"
            ),
        ),
        ("mean", does_not_raise()),
        ("worst_case", does_not_raise()),
    ],
)
@pytest.mark.parametrize(
    "equalized_odds_metric", [equalized_odds_difference, equalized_odds_ratio]
)
def test_equalized_odds_metrics_raise_on_invalid_agg(
    equalized_odds_metric: Callable, agg: str, expectation: AbstractContextManager
):
    with expectation:
        equalized_odds_metric(y_t, y_p, sensitive_features=g_1, agg=agg)


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_equal_opportunity_difference(agg_method):
    actual = equal_opportunity_difference(y_t, y_p, sensitive_features=g_1, method=agg_method)

    gm = MetricFrame(metrics=true_positive_rate, y_true=y_t, y_pred=y_p, sensitive_features=g_1)

    assert actual == gm.difference(method=agg_method)


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_equal_opportunity_difference_weighted(agg_method):
    actual = equal_opportunity_difference(
        y_t, y_p, sensitive_features=g_1, sample_weight=s_w, method=agg_method
    )

    gm = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        sample_params={"sample_weight": s_w},
    )

    assert actual == gm.difference(method=agg_method)


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_equal_opportunity_ratio(agg_method):
    actual = equal_opportunity_ratio(y_t, y_p, sensitive_features=g_1, method=agg_method)

    gm = MetricFrame(metrics=true_positive_rate, y_true=y_t, y_pred=y_p, sensitive_features=g_1)

    assert actual == gm.ratio(method=agg_method)


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_equal_opportunity_ratio_weighted(agg_method):
    actual = equal_opportunity_ratio(
        y_t, y_p, sensitive_features=g_1, sample_weight=s_w, method=agg_method
    )

    gm = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        sample_params={"sample_weight": s_w},
    )

    assert actual == gm.ratio(method=agg_method)
