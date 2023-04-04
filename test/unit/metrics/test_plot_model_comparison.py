# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from fairlearn.metrics import plot_model_comparison, make_derived_metric, selection_rate
from sklearn.metrics import accuracy_score
import numpy as np
from .data_for_test import g_1, y_p, y_t
import pytest


def test_full():
    ax = plot_model_comparison(
        x_axis_metric=accuracy_score,
        y_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds=[y_t, y_p],
        sensitive_features=g_1,
        point_labels=True,
        show_plot=False,
    )

    assert ax.get_xlabel() == "accuracy score"
    assert ax.get_ylabel() == "selection rate, difference"


def test_dict():
    plot_model_comparison(
        x_axis_metric=accuracy_score,
        y_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds={"true": y_t, "false": y_p},
        sensitive_features=g_1,
        point_labels=True,
        show_plot=False,
    )


def test_reshape():
    plot_model_comparison(
        x_axis_metric=accuracy_score,
        y_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds=y_p,
        sensitive_features=g_1,
    )


def test_wrong_shape():
    with pytest.raises(IndexError):
        plot_model_comparison(
            x_axis_metric=accuracy_score,
            y_axis_metric=make_derived_metric(
                metric=selection_rate, transform="difference"
            ),
            y_true=y_t,
            y_preds=y_p + [1],
            sensitive_features=g_1,
            point_labels=True,
            show_plot=False,
        )


def test_wrong_shape2():
    with pytest.raises(ValueError):
        plot_model_comparison(
            x_axis_metric=accuracy_score,
            y_axis_metric=make_derived_metric(
                metric=selection_rate, transform="difference"
            ),
            y_true=y_t.reshape(-1).tolist() + [1],
            y_preds=[y_p, y_t],
            sensitive_features=g_1.reshape(-1).tolist() + [1],
            point_labels=True,
            show_plot=False,
        )


def test_kw_error_labels1():
    with pytest.raises(ValueError):
        plot_model_comparison(
            x_axis_metric=accuracy_score,
            y_axis_metric=make_derived_metric(
                metric=selection_rate, transform="difference"
            ),
            y_true=y_t,
            y_preds=y_p,
            sensitive_features=g_1,
            axis_labels="true",
        )


def test_kw_error_labels2():
    with pytest.raises(ValueError):
        plot_model_comparison(
            x_axis_metric=accuracy_score,
            y_axis_metric=make_derived_metric(
                metric=selection_rate, transform="difference"
            ),
            y_true=y_t,
            y_preds=y_p,
            sensitive_features=g_1,
            axis_labels=(1, 2, 3),
        )


def test_kw_error_bools():
    with pytest.raises(ValueError):
        plot_model_comparison(
            x_axis_metric=accuracy_score,
            y_axis_metric=make_derived_metric(
                metric=selection_rate, transform="difference"
            ),
            y_true=y_t,
            y_preds=y_p,
            sensitive_features=g_1,
            show_plot="true",
            legend="true",
        )


def test_kw_error_point_labels1():
    with pytest.raises(ValueError):
        plot_model_comparison(
            x_axis_metric=accuracy_score,
            y_axis_metric=make_derived_metric(
                metric=selection_rate, transform="difference"
            ),
            y_true=y_t,
            y_preds=y_p,
            sensitive_features=g_1,
            point_labels=[1, 2, 3],
        )


def test_kw_error_point_labels2():
    with pytest.raises(ValueError):
        plot_model_comparison(
            x_axis_metric=accuracy_score,
            y_axis_metric=make_derived_metric(
                metric=selection_rate, transform="difference"
            ),
            y_true=y_t,
            y_preds=y_p,
            sensitive_features=g_1,
            point_labels="label",
        )


def test_kw_error_point_label_position():
    with pytest.raises(ValueError):
        plot_model_comparison(
            x_axis_metric=accuracy_score,
            y_axis_metric=make_derived_metric(
                metric=selection_rate, transform="difference"
            ),
            y_true=y_t,
            y_preds=y_p,
            sensitive_features=g_1,
            point_labels_position="okay",
        )


def test_kw_error_point_label_position2():
    with pytest.raises(ValueError):
        plot_model_comparison(
            x_axis_metric=accuracy_score,
            y_axis_metric=make_derived_metric(
                metric=selection_rate, transform="difference"
            ),
            y_true=y_t,
            y_preds=y_p,
            sensitive_features=g_1,
            point_labels_position=(1, "okay"),
        )


def test_other_order():
    plot_model_comparison(
        y_axis_metric=accuracy_score,
        x_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds=y_p,
        sensitive_features=g_1,
    )


def random_metric1(yp, yt):
    return np.mean(yp == yt)


def random_metric2(yp, yt):
    return np.mean(yp == yt) ** 2


def test_named():
    ax = plot_model_comparison(
        x_axis_metric=random_metric1,
        y_axis_metric=random_metric2,
        y_true=y_t,
        y_preds=y_p,
        sensitive_features=g_1,
    )

    assert ax.get_xlabel() == "random metric1"
    assert ax.get_ylabel() == "random metric2"


def test_missing_names():
    def random_metric1(yp, yt):
        return np.mean(yp == yt)

    def random_metric2(yp, yt):
        return np.mean(yp == yt) ** 2

    random_metric1.__qualname__ = ""
    random_metric1.__name__ = "test1"

    random_metric2.__qualname__ = ""
    random_metric2.__name__ = ""

    ax = plot_model_comparison(
        x_axis_metric=random_metric1,
        y_axis_metric=random_metric2,
        y_true=y_t,
        y_preds=y_p,
        sensitive_features=g_1,
    )

    assert ax.get_xlabel() == "test1"
    assert ax.get_ylabel() != ""


def test_custom_name():
    def random_metric1(yp, yt):
        return np.mean(yp == yt)

    def random_metric2(yp, yt):
        return np.mean(yp == yt) ** 2

    ax = plot_model_comparison(
        x_axis_metric=random_metric1,
        y_axis_metric=random_metric2,
        y_true=y_t,
        y_preds=y_p,
        sensitive_features=g_1,
        axis_labels=("a", "b"),
    )

    assert ax.get_xlabel() == "a"
    assert ax.get_ylabel() == "b"


def test_multiple_calls():
    ax = plot_model_comparison(
        x_axis_metric=accuracy_score,
        y_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds=[y_t],
        sensitive_features=g_1,
        point_labels=False,
        show_plot=False,
        c="red",
        label="True",
    )

    plot_model_comparison(
        ax=ax,
        x_axis_metric=accuracy_score,
        y_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds=[y_p],
        sensitive_features=g_1,
        point_labels=False,
        show_plot=False,
        c="blue",
        label="Pred",
        legend=True,
    )

    assert ax.get_xlabel() == "accuracy score"
    assert ax.get_ylabel() == "selection rate, difference"
    assert ax.get_legend_handles_labels()[1] == ["True", "Pred"]
