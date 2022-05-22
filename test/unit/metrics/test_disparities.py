# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pytest

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)

from .data_for_test import g_1, s_w, y_p, y_t

_aggregate_methods = ["between_groups", "to_overall"]


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_demographic_parity_difference(agg_method):
    actual = demographic_parity_difference(
        y_t, y_p, sensitive_features=g_1, method=agg_method
    )

    gm = MetricFrame(
        metrics=selection_rate, y_true=y_t, y_pred=y_p, sensitive_features=g_1
    )

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
    actual = demographic_parity_ratio(
        y_t, y_p, sensitive_features=g_1, method=agg_method
    )

    gm = MetricFrame(
        metrics=selection_rate, y_true=y_t, y_pred=y_p, sensitive_features=g_1
    )

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


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_equalized_odds_difference(agg_method):
    actual = equalized_odds_difference(
        y_t, y_p, sensitive_features=g_1, method=agg_method
    )

    metrics = {"tpr": true_positive_rate, "fpr": false_positive_rate}
    gm = MetricFrame(metrics=metrics, y_true=y_t, y_pred=y_p, sensitive_features=g_1)

    diffs = gm.difference(method=agg_method)
    assert actual == diffs.max()


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_equalized_odds_difference_weighted(agg_method):
    actual = equalized_odds_difference(
        y_t, y_p, sensitive_features=g_1, method=agg_method, sample_weight=s_w
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
    assert actual == diffs.max()


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_equalized_odds_ratio(agg_method):
    actual = equalized_odds_ratio(y_t, y_p, method=agg_method, sensitive_features=g_1)

    metrics = {"tpr": true_positive_rate, "fpr": false_positive_rate}
    gm = MetricFrame(metrics=metrics, y_true=y_t, y_pred=y_p, sensitive_features=g_1)

    ratios = gm.ratio(method=agg_method)
    assert actual == ratios.min()


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_equalized_odds_ratio_weighted(agg_method):
    actual = equalized_odds_ratio(
        y_t, y_p, method=agg_method, sensitive_features=g_1, sample_weight=s_w
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
    assert actual == ratios.min()
