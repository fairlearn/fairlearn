# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import pytest
import sklearn.metrics as skm

import fairlearn.metrics as metrics

# Synthetic dataset
y_true = [0, 1, 0, 0, 1]
y_pred = [0, 1, 1, 0, 1]
sf = ["a", "a", "b", "b", "b"]
accuracy_score_difference = 1.0 / 3.0

# Removed in version:
version = "0.10.0"


def test_all_positional_arguments():
    msg = (
        "You have provided 'metrics', 'y_true', 'y_pred' as positional arguments."
        f" Please pass them as keyword arguments. From version {version} passing them"
        " as positional arguments will result in an error."
    )

    with pytest.warns(FutureWarning) as record:
        mf = metrics.MetricFrame(
            skm.accuracy_score, y_true, y_pred, sensitive_features=sf
        )

    assert len(record) == 1
    assert str(record[0].message) == msg
    assert mf.difference() == pytest.approx(accuracy_score_difference)


def test_one_positional_argument():
    msg = (
        "You have provided 'metrics' as positional arguments. Please pass them as"
        f" keyword arguments. From version {version} passing them as positional"
        " arguments will result in an error."
    )

    with pytest.warns(FutureWarning) as record:
        mf = metrics.MetricFrame(
            skm.accuracy_score, y_true=y_true, y_pred=y_pred, sensitive_features=sf
        )

    assert len(record) == 1
    assert str(record[0].message) == msg
    assert mf.difference() == pytest.approx(accuracy_score_difference)


def test_four_positional_arguments():
    # The first formal positional argument to the constructor is "self"
    # so the error message says that five arguments were given
    msg = "__init__() takes 1 positional argument but 5 positional arguments were given"

    with pytest.raises(TypeError) as execInfo:
        _ = metrics.MetricFrame(skm.accuracy_score, y_true, y_pred, sf)

    assert execInfo.value.args[0] == msg


def test_keyword_metric():
    msg = (
        "The positional argument 'metric' has been replaced by "
        f"a keyword argument 'metrics'. From version {version} passing "
        "it as a positional argument or as a keyword argument "
        "'metric' will result in an error"
    )

    with pytest.warns(FutureWarning) as record:
        mf = metrics.MetricFrame(
            metric=skm.accuracy_score,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sf,
        )

    assert len(record) == 1
    assert str(record[0].message) == msg
    assert mf.difference() == pytest.approx(accuracy_score_difference)


def test_no_warnings(recwarn):
    mf = metrics.MetricFrame(
        metrics=skm.accuracy_score,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sf,
    )

    assert len(recwarn) == 0
    assert mf.difference() == pytest.approx(accuracy_score_difference)


def test_metric_and_metrics():
    warn_msg = (
        "The positional argument 'metric' has been replaced by "
        f"a keyword argument 'metrics'. From version {version} passing "
        "it as a positional argument or as a keyword argument "
        "'metric' will result in an error"
    )
    error_msg = "__init__() got multiple values for keyword argument 'metrics'"

    with pytest.warns(FutureWarning) as warn_record:
        with pytest.raises(TypeError) as error_execInfo:
            _ = metrics.MetricFrame(
                metric=skm.accuracy_score,
                metrics=skm.accuracy_score,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sf,
            )

    assert len(warn_record) == 1
    assert str(warn_record[0].message) == warn_msg
    assert error_execInfo.value.args[0].endswith(error_msg)
