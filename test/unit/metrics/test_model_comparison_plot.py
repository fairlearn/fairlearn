# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from fairlearn.metrics import plot_model_comparison, make_derived_metric, selection_rate
from sklearn.metrics import accuracy_score

from .data_for_test import g_1, y_p, y_t


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
    ax = plot_model_comparison(
        x_axis_metric=accuracy_score,
        y_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds={'true':y_t, 'false':y_p},
        sensitive_features=g_1,
        point_labels=True,
        show_plot=False,
    )

def test_reshape():
    ax = plot_model_comparison(
        x_axis_metric=accuracy_score,
        y_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds=y_p,
        sensitive_features=g_1,
        point_labels=True,
        show_plot=False,
    )

def test_wrong_shape():
    ax = plot_model_comparison(
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


def test_kw_error_labels1():
    ax = plot_model_comparison(
        x_axis_metric=accuracy_score,
        y_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds=y_p,
        sensitive_features=g_1,
        axis_labels="true"
    )

def test_kw_error_labels2():
    ax = plot_model_comparison(
        x_axis_metric=accuracy_score,
        y_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds=y_p,
        sensitive_features=g_1,
        axis_labels=(1, 2, 3)
    )


def test_kw_error_bools():
    ax = plot_model_comparison(
        x_axis_metric=accuracy_score,
        y_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds=y_p,
        sensitive_features=g_1,
        show_plot="true",
        legend="true"
    )


def test_kw_error_point_labels1():
    ax = plot_model_comparison(
        x_axis_metric=accuracy_score,
        y_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds=y_p,
        sensitive_features=g_1,
        point_labels=[1,2,3]
    )


def test_kw_error_point_labels2():
    ax = plot_model_comparison(
        x_axis_metric=accuracy_score,
        y_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds=y_p,
        sensitive_features=g_1,
        point_labels="label"
    )


def test_kw_error_point_label_position():
    ax = plot_model_comparison(
        x_axis_metric=accuracy_score,
        y_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds=y_p,
        sensitive_features=g_1,
        point_labels_position="okay"
    )


def test_kw_error_point_label_position2():
    ax = plot_model_comparison(
        x_axis_metric=accuracy_score,
        y_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds=y_p,
        sensitive_features=g_1,
        point_labels_position=(1, "okay")
    )


def test_kw_error_point_legend_kws():
    ax = plot_model_comparison(
        x_axis_metric=accuracy_score,
        y_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds=y_p,
        sensitive_features=g_1,
        legend_kwargs=1
    )









def test_other_order():
    ax = plot_model_comparison(
        y_axis_metric=accuracy_score,
        x_axis_metric=make_derived_metric(
            metric=selection_rate, transform="difference"
        ),
        y_true=y_t,
        y_preds=y_p,
        sensitive_features=g_1,
    )


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
