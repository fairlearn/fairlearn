# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pytest

from fairlearn.metrics import MetricFrame

# Group "b" gets a NaN metric value; groups "a" and "c" do not.
_Y_TRUE = [0, 1, 0, 1, 1, 1]
_Y_PRED = [1, 1, 0, 1, 1, 1]
_SF = ["a", "a", "b", "b", "c", "c"]


def _nan_for_b(y_true, y_pred):
    if list(y_true) == [0, 1] and list(y_pred) == [0, 1]:
        return np.nan
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _none_for_b(y_true, y_pred):
    value = _nan_for_b(y_true, y_pred)
    return None if np.isnan(value) else value


def _accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


@pytest.mark.parametrize(
    "aggregate",
    [
        lambda mf: mf.group_min(),
        lambda mf: mf.group_max(),
        lambda mf: mf.difference(),
        lambda mf: mf.difference(method="to_overall"),
        lambda mf: mf.ratio(),
        lambda mf: mf.ratio(method="to_overall"),
    ],
    ids=[
        "group_min",
        "group_max",
        "difference",
        "difference_to_overall",
        "ratio",
        "ratio_to_overall",
    ],
)
@pytest.mark.parametrize("metric", [_nan_for_b, _none_for_b])
def test_aggregates_warn_when_they_leave_out_a_nan_group(aggregate, metric):
    mf = MetricFrame(metrics=metric, y_true=_Y_TRUE, y_pred=_Y_PRED, sensitive_features=_SF)

    with pytest.warns(UserWarning, match=r"left out these groups.*\['b'\]"):
        aggregate(mf)


def test_left_out_groups_do_not_change_the_aggregates():
    mf = MetricFrame(metrics=_nan_for_b, y_true=_Y_TRUE, y_pred=_Y_PRED, sensitive_features=_SF)

    with pytest.warns(UserWarning):
        assert mf.group_min() == 0.5
    with pytest.warns(UserWarning):
        assert mf.group_max() == 1.0
    with pytest.warns(UserWarning):
        assert mf.difference() == 0.5
    with pytest.warns(UserWarning):
        assert mf.ratio() == 0.5


def test_the_warning_names_only_the_metrics_with_nan_groups():
    mf = MetricFrame(
        metrics={"nan_for_b": _nan_for_b, "accuracy": _accuracy},
        y_true=_Y_TRUE,
        y_pred=_Y_PRED,
        sensitive_features=_SF,
    )

    with pytest.warns(UserWarning) as record:
        mf.difference()

    (warning,) = record
    assert "'nan_for_b': ['b']" in str(warning.message)
    assert "accuracy" not in str(warning.message)


def test_by_group_and_overall_do_not_warn():
    mf = MetricFrame(metrics=_nan_for_b, y_true=_Y_TRUE, y_pred=_Y_PRED, sensitive_features=_SF)

    assert np.isnan(mf.by_group["b"])
    assert mf.overall == pytest.approx(5 / 6)


def test_combinations_that_never_occur_do_not_warn():
    # ("b", "y") never occurs, so it is NaN by construction, not a metric value.
    mf = MetricFrame(
        metrics=_accuracy,
        y_true=[0, 1, 0, 1],
        y_pred=[1, 1, 0, 1],
        sensitive_features=["a", "a", "b", "b"],
        control_features=["x", "y", "x", "x"],
    )

    assert np.isnan(mf.by_group[("y", "b")])
    mf.group_min()
    mf.difference()
    mf.ratio()


def test_aggregates_without_nan_groups_do_not_warn():
    mf = MetricFrame(metrics=_accuracy, y_true=_Y_TRUE, y_pred=_Y_PRED, sensitive_features=_SF)

    mf.group_min()
    mf.group_max()
    mf.difference()
    mf.ratio()


def test_none_is_left_out_and_named_but_a_non_scalar_is_not():
    # Group "a" returns an array, which errors="coerce" handles; group "b" returns None.
    def metric(y_true, y_pred):
        groups = {
            ((0, 1), (1, 1)): np.array([1, 2]),
            ((0, 1), (0, 1)): None,
        }
        return groups.get((tuple(y_true), tuple(y_pred)), 1.0)

    mf = MetricFrame(metrics=metric, y_true=_Y_TRUE, y_pred=_Y_PRED, sensitive_features=_SF)

    assert mf.by_group["b"] is None
    with pytest.warns(UserWarning, match=r"\{'metric': \['b'\]\}"):
        assert mf.difference(errors="coerce") == 0
