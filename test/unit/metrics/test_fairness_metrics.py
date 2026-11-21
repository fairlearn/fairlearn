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


# Group "a" has no positive labels and group "c" has no negative labels, and every
# prediction is correct, so every defined rate is equal across groups.
_y_true_missing_class = [0, 0, 0, 1, 1, 0, 1, 1]
_sf_missing_class = ["a", "a", "b", "b", "b", "b", "c", "c"]


@pytest.mark.parametrize("agg_method", _aggregate_methods)
@pytest.mark.parametrize("agg", _agg_options)
def test_equalized_odds_leaves_out_groups_with_an_undefined_rate(agg_method, agg):
    with pytest.warns(UserWarning) as record:
        difference = equalized_odds_difference(
            _y_true_missing_class,
            _y_true_missing_class,
            sensitive_features=_sf_missing_class,
            method=agg_method,
            agg=agg,
        )
        ratio = equalized_odds_ratio(
            _y_true_missing_class,
            _y_true_missing_class,
            sensitive_features=_sf_missing_class,
            method=agg_method,
            agg=agg,
        )

    assert difference == 0
    assert ratio == 1
    messages = [str(warning.message) for warning in record]
    assert any("true positive rate" in m and "['a']" in m for m in messages)
    assert any("false positive rate" in m and "['c']" in m for m in messages)


@pytest.mark.parametrize("agg_method", _aggregate_methods)
def test_equal_opportunity_leaves_out_groups_with_no_positive_labels(agg_method):
    with pytest.warns(UserWarning, match=r"true positive rate .* \['a'\]"):
        difference = equal_opportunity_difference(
            _y_true_missing_class,
            _y_true_missing_class,
            sensitive_features=_sf_missing_class,
            method=agg_method,
        )
    with pytest.warns(UserWarning, match=r"true positive rate .* \['a'\]"):
        ratio = equal_opportunity_ratio(
            _y_true_missing_class,
            _y_true_missing_class,
            sensitive_features=_sf_missing_class,
            method=agg_method,
        )

    assert difference == 0
    assert ratio == 1


def test_equalized_odds_still_measures_the_groups_whose_rates_are_defined():
    # Group "b" misses one of its two positives and flags one of its two negatives.
    y_pred = [0, 0, 1, 1, 0, 0, 1, 1]
    with pytest.warns(UserWarning):
        difference = equalized_odds_difference(
            _y_true_missing_class, y_pred, sensitive_features=_sf_missing_class
        )
    # TPR: b 0.5, c 1.0. FPR: a 0.0, b 0.5. The worst case is 0.5.
    assert difference == pytest.approx(0.5)


def test_equalized_odds_treats_zero_weight_labels_as_absent():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 1]
    with pytest.warns(UserWarning, match=r"true positive rate .* \['a'\]"):
        difference = equalized_odds_difference(
            y_true, y_pred, sensitive_features=["a", "a", "b", "b"], sample_weight=[1, 0, 1, 1]
        )

    assert difference == 0


@pytest.mark.parametrize("y_true", [[1, 1], [0, 0]])
def test_equalized_odds_uses_the_rate_that_is_defined(y_true):
    with pytest.warns(UserWarning):
        difference = equalized_odds_difference(y_true, [1, 0], sensitive_features=["a", "b"])
    with pytest.warns(UserWarning):
        ratio = equalized_odds_ratio(y_true, [1, 0], sensitive_features=["a", "b"])

    assert difference == 1
    assert ratio == 0


def test_equal_opportunity_is_nan_when_no_group_has_a_defined_rate():
    with pytest.warns(UserWarning):
        ratio = equal_opportunity_ratio([0, 0], [1, 0], sensitive_features=["a", "b"])

    assert ratio != ratio  # NaN


def test_equalized_odds_does_not_warn_when_every_rate_is_defined():
    difference = equalized_odds_difference(y_t, y_p, sensitive_features=g_1)

    assert difference == max(
        MetricFrame(
            metrics={"tpr": true_positive_rate, "fpr": false_positive_rate},
            y_true=y_t,
            y_pred=y_p,
            sensitive_features=g_1,
        ).difference()
    )
