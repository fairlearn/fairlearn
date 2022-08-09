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
